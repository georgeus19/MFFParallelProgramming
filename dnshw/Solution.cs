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

            AddrSubdomainPair cachedSubdomainServer = FindCachedSubdomain(domains).Result;

			IP4Addr res;
			if (cachedSubdomainServer.DomainCount == 0) {
				//Console.WriteLine("DomainCount 0");
				res = dnsClient.GetRootServers()[0];
			} else {
                //Console.WriteLine(cachedSubdomainServer.Addr.Result);
                res = cachedSubdomainServer.Addr.Result;
            }

			if (cachedSubdomainServer.DomainCount == domains.Length) {
				return cachedSubdomainServer.Addr;
            }

			Task<IP4Addr> t = dnsClient.Resolve(res, domains[cachedSubdomainServer.DomainCount]);
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

            return t;
        }


		private Task<AddrSubdomainPair> FindCachedSubdomain(IEnumerable<string> domains) {
			List<Task<AddrSubdomainPair>> cachedAddrs = new();

			foreach (string subdomain in GenerateSubdomains(domains)) {
				bool inCache = _cache.TryGetValue(subdomain, out AddrSubdomainPair possibleAddr);
				if (inCache) {
				//Console.WriteLine($"subdomain: {subdomain}");
					if (possibleAddr.IsCompletedSuccessfully) {
						cachedAddrs.Add(CheckForCachedSubdomain(subdomain, possibleAddr));
                    } else if (possibleAddr.IsCompleted) {
						//Console.WriteLine($"FindCachedSubdomain: IsCompleted");
						continue;
                    } else {
						cachedAddrs.Add(Task.FromResult(possibleAddr));
                    }
                    //Console.WriteLine($"FindCachedSubdomain: {subdomain}");
				}
			}

			return Task.WhenAll(cachedAddrs).ContinueWith<AddrSubdomainPair>((t) => {
				int maxDomainCount = t.Result.DefaultIfEmpty().Max(AddrSubdomainPair => AddrSubdomainPair.DomainCount);
				return t.Result.Where((p) => p.DomainCount == maxDomainCount).FirstOrDefault();
			});
        }


		private Task<AddrSubdomainPair> CheckForCachedSubdomain(string subdomain, AddrSubdomainPair addr) {

			return dnsClient.Reverse(addr.Addr.Result).ContinueWith<AddrSubdomainPair>((t) => {
				if (t.IsCompletedSuccessfully && t.Result == subdomain) {
					return addr;
                } else {
					return new AddrSubdomainPair();
                }
			});

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

   //     private Task<IP4Addr> GetDomainServer(string subDomain, IP4Addr server) {
   //         bool inCache = _cache.TryGetValue(subDomain, out IP4Addr addr);

			//Task<IP4Addr> task = null;

   //         if (inCache) {
			//	task = dnsClient.Reverse(addr).ContinueWith<IP4Addr>((t) => {
			//		if (!t.IsFaulted && subDomain == t.Result) {
			//			return addr;
			//		}
			//		else {
			//			throw new Exception();
			//		}

			//	}).ContinueWith<IP4Addr>(async (t) => {
			//		if (t.IsFaulted) {
			//			return await dnsClient.Resolve(server, subDomain);

			//		}
			//		return await t;
   //             });

			//	task.Wait();
			//	if (task.IsFaulted) {
			//		return dnsClient.Resolve(server, subDomain);
   //             } else {
			//		return task;
   //             }
   //         } else {
			//	return dnsClient.Resolve(server, subDomain);
			//}

        //}
    }


	//class DNSCache {
	//	private ConcurrentDictionary<string, IP4Addr> _cache;
	//	private IDNSClient _dnsClient;

	//	public DNSCache(IDNSClient dNSClient) {
	//		_cache = new ConcurrentDictionary<string, IP4Addr>();
	//		_dnsClient = dNSClient;

	//	}

	//	public bool Get(string key, out IP4Addr resultAddr) {
	//		bool inCache = _cache.TryGetValue(key, out IP4Addr addr);

 //           /*if (!inCache) {
 //               resultAddr = IP4Addr();
 //               return false;
 //           }

 //           var t = _dnsClient.Reverse(addr);
 //           t.ContinueWith<IP4Addr>(t => {
 //               if (!t.IsFaulted && key == t.Result) {
 //                   return addr;
 //               }
 //               else {
 //                   _dnsClient.Resolve()

 //               }
 //           });*/


 //       }


    //}
}
