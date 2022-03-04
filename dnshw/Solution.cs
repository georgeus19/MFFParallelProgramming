using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Text;
using System.Threading.Tasks;
using System.Linq;

namespace dns_netcore {


    struct AddrSubdomainPair {
        public Task<IP4Addr> Addr { get; }
		public int DomainCount { get; }

		public bool IsCompletedSuccessfully => Addr.IsCompletedSuccessfully;
		public bool IsCompleted => Addr.IsCompleted;

		public AddrSubdomainPair(Task<IP4Addr> addr, int domainCount) {
			this.Addr = addr;
			this.DomainCount = domainCount;
		}

		public AddrSubdomainPair(IP4Addr addr, int domainCount) {
			this.Addr = Task.FromResult(addr);
			this.DomainCount = domainCount;
		}

		public AddrSubdomainPair() {
			this.Addr = Task.FromResult(new IP4Addr());
			this.DomainCount = 0;
        }
    }

    class RecursiveResolver : IRecursiveResolver
	{
		private IDNSClient dnsClient;
		private ConcurrentDictionary<string, AddrSubdomainPair> _cache;
		public RecursiveResolver(IDNSClient client)
		{
			this.dnsClient = client;
			_cache = new ConcurrentDictionary<string, AddrSubdomainPair>();
		}

		public Task<IP4Addr> ResolveRecursive(string domain)
		{
            //Console.WriteLine($"{domain} cache: {_cache.Count()}");
            string[] domains = domain.Split('.');
			Array.Reverse(domains);

            AddrSubdomainPair cachedSubdomainServer = FindCachedSubdomain(domains);

			Task<IP4Addr> res;
			if (cachedSubdomainServer.DomainCount == 0) {
                //Console.WriteLine("DomainCount 0");
                res = Task.FromResult(dnsClient.GetRootServers()[0]);
			} else {
                //Console.WriteLine($"DomainCount {cachedSubdomainServer.DomainCount}");
                res = cachedSubdomainServer.Addr;
            }

			if (cachedSubdomainServer.DomainCount == domains.Length) {
				return cachedSubdomainServer.Addr;
            }

			Task<IP4Addr> t = res.ContinueWith<Task<IP4Addr>>((t) => {
				return dnsClient.Resolve(t.Result, domains[cachedSubdomainServer.DomainCount]);
			}).Unwrap();
			string subdomain = string.Join(".", domains.Take(cachedSubdomainServer.DomainCount + 1));
			_cache[subdomain] = new AddrSubdomainPair(t, cachedSubdomainServer.DomainCount + 1);	


			for (int i = cachedSubdomainServer.DomainCount + 1; i < domains.Length; ++i) {
				string d = domains[i];
				t = t.ContinueWith<Task<IP4Addr>>(task => {
					return dnsClient.Resolve(task.Result, d);
				}).Unwrap();
				subdomain = string.Join(".", domains.Take(i + 1));
				_cache[subdomain] = new AddrSubdomainPair(t, i + 1);
			}
            //Console.WriteLine("Cache:");
            //foreach (var k in _cache) {
            //    Console.WriteLine($"{k.Key}, {k.Value}");
            //}
            //Console.WriteLine("Cache END!");
            return t;
        }


		private AddrSubdomainPair FindCachedSubdomain(string[] domains) {
			Array.Reverse(domains);
			string domain = string.Join('.', domains);
			Array.Reverse(domains);

			while (domain != "") {
                //Console.WriteLine($"doamin {domain}");
				int dotIndex = domain.IndexOf('.');
				if (dotIndex < 0) {
					return new AddrSubdomainPair();
                }
				domain = domain.Substring(dotIndex + 1);
                //Console.WriteLine($"DOMAINS {domain}");
				string tmp = string.Join('.', domain.Split('.').Reverse().ToArray());
				//Console.WriteLine($"TMP {tmp}");

				bool inCache = _cache.TryGetValue(tmp, out AddrSubdomainPair possibleAddr);
				if (inCache) {
					//Console.WriteLine($"subdomain: {subdomain}");
					if (!possibleAddr.IsCompleted) {
						return possibleAddr;
					}
					else if (possibleAddr.IsCompletedSuccessfully) {

						return CheckForCachedSubdomain(domain, possibleAddr);

					}
				}
			}


			//foreach (string subdomain in GenerateSubdomains(domains)) {
			//	bool inCache = _cache.TryGetValue(subdomain, out AddrSubdomainPair possibleAddr);
			//	if (inCache) {
			//	//Console.WriteLine($"subdomain: {subdomain}");
			//		if (!possibleAddr.IsCompleted) {
			//			return possibleAddr;
   //                 } else if (possibleAddr.IsCompletedSuccessfully) {
			//			return CheckForCachedSubdomain(subdomain, possibleAddr);

   //                 }
			//	}
			//}
			return new AddrSubdomainPair();
        }


		private AddrSubdomainPair CheckForCachedSubdomain(string subdomain, AddrSubdomainPair addr) {
			
			Task<IP4Addr> task = dnsClient.Reverse(addr.Addr.Result).ContinueWith<Task<IP4Addr>>((t) => {
				if (t.IsCompletedSuccessfully && t.Result == subdomain) {
					return addr.Addr;
                } else {
					var ddd = subdomain.Split('.');
                    Array.Reverse(ddd);
                    var tt = ResolveRecursive(string.Join('.', ddd));
					_cache[subdomain] = new AddrSubdomainPair(tt, addr.DomainCount);
					return tt;
                }
			}).Unwrap();

			return new AddrSubdomainPair(task, addr.DomainCount);
        }

		private IEnumerable<string> GenerateSubdomains(IEnumerable<string> domains) {
			StringBuilder sb = new();
			bool first = true;

			foreach (string domain in domains) {
				if (first) {
					sb.Append(domain);
					first = false;
				} else {
					sb.Append('.');
					sb.Append(domain);

				}
				yield return sb.ToString();
			}
		}

    }
}
