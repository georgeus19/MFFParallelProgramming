using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Text;
using System.Threading.Tasks;
using System.Linq;

namespace dns_netcore {

    struct AddrSubdomainPair {
        public Task<IP4Addr> Addr { get; }
		public string Domain { get; }

		public bool IsCompletedSuccessfully => Addr.IsCompletedSuccessfully;
		public bool IsCompleted => Addr.IsCompleted;

		public AddrSubdomainPair(Task<IP4Addr> addr, string domain) {
			this.Addr = addr;
			this.Domain = domain;
		}

		public AddrSubdomainPair(IP4Addr addr, string domain) {
			this.Addr = Task.FromResult(addr);
			this.Domain = domain;
		}

		public AddrSubdomainPair() {
			this.Addr = Task.FromResult(new IP4Addr());
			this.Domain = string.Empty;
		}
	}

    class RecursiveResolver : IRecursiveResolver
	{
		private IDNSClient _dnsClient;
		private ConcurrentDictionary<string, AddrSubdomainPair> _cache;
		private int _lastUsedRoot = 0;
		private object _l = new();

		public RecursiveResolver(IDNSClient client)	{
			this._dnsClient = client;
			_cache = new ConcurrentDictionary<string, AddrSubdomainPair>();
		}

		private IP4Addr GetRootServer() {
			lock(_l) {
				var roots = _dnsClient.GetRootServers();
				return roots[++_lastUsedRoot % roots.Count];
            }
        }

		private string ComputeLeftoverDomainPartToResolve(string domain, string resolvedDomainPart) {
            if (resolvedDomainPart == string.Empty) {
                return domain;
            }

            int i = domain.LastIndexOf(resolvedDomainPart);
            if (domain[i - 1] == '.') {
                --i;
            }
            return domain[..i];



            if (domain == resolvedDomainPart) {
                return string.Empty;
            }

            return domain[..(domain.LastIndexOf(resolvedDomainPart) - 1)];
        }

		private string AddSubdomain(string domain, string subdomain) {
			if (domain == string.Empty) {
				return subdomain;
            } else {
				return subdomain + '.' + domain;
            }

        }

		public Task<IP4Addr> ResolveRecursive(string domain) {
            //Console.WriteLine($"ResolveRecursive start: {domain}");
            AddrSubdomainPair asp = FindCachedSubdomain(domain);

			if (asp.Domain == domain) {
				return asp.Addr;
			}

			if (asp.Domain == string.Empty) {
				asp = new AddrSubdomainPair(Task.FromResult(GetRootServer()), string.Empty);
			}

			//Console.WriteLine($"asp.Domain {asp.Domain}");

			//Console.WriteLine($"needToResolveDomain: '{needToResolveDomain}'");

			string domainToCache = asp.Domain;

			string domainPartToResolve = ComputeLeftoverDomainPartToResolve(domain, asp.Domain);

			foreach (string subdomain in domainPartToResolve.Split('.').Reverse()) {
                Console.WriteLine(subdomain);
				Task<IP4Addr> addr = asp.Addr.ContinueWith((t) => {
					return _dnsClient.Resolve(t.Result, subdomain);
				}).Unwrap();
				domainToCache = AddSubdomain(domainToCache, subdomain);
				asp = new AddrSubdomainPair(addr, domainToCache);
				_cache[asp.Domain] = asp;
			}

            Console.WriteLine("Cahce start:");
            foreach (var k in _cache) {
                Console.WriteLine($"	Key {k.Key}");
            }
            _cache[asp.Domain] = asp;

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
			
			Task<IP4Addr> task = _dnsClient.Reverse(addr.Addr.Result).ContinueWith<Task<IP4Addr>>((t) => {
				if (t.IsCompletedSuccessfully && t.Result == domain) {
					return addr.Addr;
                } else {
                    _cache.Remove(domain, out var _);
                    var tt = ResolveRecursive(domain);
                    _cache[domain] = new AddrSubdomainPair(tt, domain);
                    return tt;
                }
            }).Unwrap();

			return new AddrSubdomainPair(task, domain);
        }
    }
}
