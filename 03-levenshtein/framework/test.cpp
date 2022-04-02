#define _CRT_SECURE_NO_WARNINGS

/*
 * Levenshtein's Edit Distance
 */

#include <exception.hpp>
#include <stopwatch.hpp>
#include <interface.hpp>
#include <implementation.hpp>

#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <cstdint>
#include <cstdio>


typedef std::uint32_t char_t;


template<bool DEBUG>
std::size_t computeDistance(const std::vector<char_t> str1, const std::vector<char_t> &str2)
{
	// Initialize distance functor.
	EditDistance<char_t, std::size_t, DEBUG> distance;
	distance.init(str1.size(), str2.size());

	// Compute the distance.
	std::size_t res = distance.compute(str1, str2);

	return res;
}


int main(int argc, char **argv)
{
	std::vector<char_t> a {'s', 'a', 't', 'u', 'r', 'd', 'a', 'y'};
	std::vector<char_t> b {'s', 'u', 'n', 'd', 'a', 'y'};

    std::cout << "Exptected distance: 3" << std::endl;
    auto dist = computeDistance<true>(a, b);
    std::cout << "lev(saturday, sunday)" << dist << std::endl;
    // dist = computeDistance<true>(b, a);
    // std::cout << "lev(sunday, saturday)" << dist << std::endl;

	return 0;
}
