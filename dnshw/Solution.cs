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

		public string Domain { get; }

		public bool IsCompletedSuccessfully => Addr.IsCompletedSuccessfully;
		public bool IsCompleted => Addr.IsCompleted;

		public AddrSubdomainPair(Task<IP4Addr> addr, int domainCount, string domain) {
			this.Addr = addr;
			this.DomainCount = domainCount;
			this.Domain = domain;
		}

		public AddrSubdomainPair(IP4Addr addr, int domainCount, string domain) {
			this.Addr = Task.FromResult(addr);
			this.DomainCount = domainCount;
			this.Domain = domain;
		}

		public AddrSubdomainPair() {
			this.Addr = Task.FromResult(new IP4Addr());
			this.DomainCount = 0;
			this.Domain = string.Empty;
		}
	}

    class RecursiveResolver : IRecursiveResolver
	{
		private IDNSClient dnsClient;
		private ConcurrentDictionary<string, AddrSubdomainPair> _cache;
		private int _lastUsedRoot = 0;
		public RecursiveResolver(IDNSClient client)
		{
			this.dnsClient = client;
			_cache = new ConcurrentDictionary<string, AddrSubdomainPair>();
		}

		private IP4Addr OP_LAD_BALANDER() {

			var roots = dnsClient.GetRootServers();
			return roots[++_lastUsedRoot % roots.Count];
			
        }

		public Task<IP4Addr> ResolveRecursive(string domain) {
            //Console.WriteLine($"ResolveRecursive start: {domain}");
            AddrSubdomainPair asp = FindCachedSubdomain(domain);

			if (asp.Domain == domain) {
				return asp.Addr;
			}

			if (asp.Domain == string.Empty) {
				asp = new AddrSubdomainPair(Task.FromResult(OP_LAD_BALANDER()), 0, string.Empty);
			}

            //Console.WriteLine($"asp.Domain {asp.Domain}");

			int i = domain.LastIndexOf(asp.Domain);
			if (i < domain.Length && domain[i - 1] == '.') {
				--i;
            }

			string needToResolveDomain = domain.Substring(0, i);
            //Console.WriteLine($"needToResolveDomain: '{needToResolveDomain}'");

			string s = asp.Domain;
			if (asp.Domain != string.Empty) {
				s = "." + s;
			}
			
			foreach(string d in needToResolveDomain.Split('.').Reverse()) {
				Task<IP4Addr> addr = asp.Addr.ContinueWith((t) => {
					return dnsClient.Resolve(t.Result, d);
				}).Unwrap();
				if (s == string.Empty) {
					s = d;
                } else {
					s = d + '.' + s;
				}
				asp = new AddrSubdomainPair(addr, 0, s);
				_cache[asp.Domain] = asp;
			}

   //         Console.WriteLine("Cahce start:");
			//foreach(var k in _cache) {
   //             Console.WriteLine($"	Key {k.Key}");
   //         }
			//_cache[asp.Domain] = asp;

			return asp.Addr;
		}

		private AddrSubdomainPair FindCachedSubdomain(string domain) {
			int dotIndex;
			do {
				//Console.WriteLine($"Query cache for domain: {domain}");
				bool inCache = _cache.TryGetValue(domain, out AddrSubdomainPair possibleAddr);
				if (inCache) {
                    //Console.WriteLine($"subdomain: {subdomain}");

                    if (!possibleAddr.IsCompleted) {
                        return possibleAddr;
                    }
                    else if (possibleAddr.IsCompletedSuccessfully) {
                        return CheckForCachedSubdomain(domain, possibleAddr);
                    }
                }
				dotIndex = domain.IndexOf('.');
				domain = domain[(dotIndex + 1)..];

			} while (dotIndex >= 0);
			
			return new AddrSubdomainPair();
        }


		private AddrSubdomainPair CheckForCachedSubdomain(string domain, AddrSubdomainPair addr) {
            //Console.WriteLine($"CheckForCachedSubdomain {domain}");
			
			Task<IP4Addr> task = dnsClient.Reverse(addr.Addr.Result).ContinueWith<Task<IP4Addr>>((t) => {
				if (t.IsCompletedSuccessfully && t.Result == domain) {
					return addr.Addr;
                } else {
                    int dotIndex = domain.IndexOf('.');
                    if (dotIndex < 0) {
						var tt = dnsClient.Resolve(OP_LAD_BALANDER(), domain);
						_cache[domain] = new AddrSubdomainPair(tt, addr.DomainCount, domain);
						return tt;
					} else {
						string subdomain = domain[(dotIndex + 1)..];
						string firstDomain = domain[..dotIndex];
						var tt = ResolveRecursive(subdomain).ContinueWith((t) => {
							return dnsClient.Resolve(t.Result, firstDomain);
                        }).Unwrap();
						_cache[domain] = new AddrSubdomainPair(tt, addr.DomainCount, domain);
						return tt;
					}
                    //_cache.Remove(domain, out var _);
                    //var tt = ResolveRecursive(domain);
                    //_cache[domain] = new AddrSubdomainPair(tt, addr.DomainCount, domain);
                    //return tt;
                }
			}).Unwrap();

			return new AddrSubdomainPair(task, addr.DomainCount, domain);
        }

    }
}
