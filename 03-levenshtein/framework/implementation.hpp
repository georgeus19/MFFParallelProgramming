#ifndef LEVENSHTEIN_IMPLEMENTATION_HPP
#define LEVENSHTEIN_IMPLEMENTATION_HPP

#include <vector>
#include <algorithm>
#include <iostream>
#include <tuple>

template<typename C = char, typename DIST = std::size_t, bool DEBUG = false>
class EditDistance : public IEditDistance<C, DIST, DEBUG> {
public:

	int qwe = 42;
	/*
	 * \brief Perform the initialization of the functor (e.g., allocate memory buffers).
	 * \param len1, len2 Lengths of first and second string respectively.
	 */
	virtual void init(DIST len1, DIST len2) {
		_currentStripe.resize((std::size_t)std::min<DIST>(len1, len2) + 2);
		_previousStripe.resize((std::size_t)std::min<DIST>(len1, len2) + 2);
		_previousPreviousStripe.resize((std::size_t)std::min<DIST>(len1, len2) + 2);
	}


	/*
	 * \brief Compute the distance between two strings.
	 * \param str1, str2 Strings to be compared.
	 * \result The computed edit distance.
	 */
	virtual DIST compute(const std::vector<C> &str1, const std::vector<C> &str2) {
		if (str1.size() <= str2.size()) {
			return lev(str1, str2);
		} else {
			return lev(str2, str1);
		}
	}

	~EditDistance() {}
private:
	std::vector<DIST> _currentStripe;
	std::vector<DIST> _previousStripe;
	std::vector<DIST> _previousPreviousStripe;

	DIST lev(const std::vector<C>& a, const std::vector<C>& b) { // a.len <= b.len
		std::size_t currentStripeStart = 1;
		std::size_t currentStripeEnd = 1;

		_previousStripe[0] = 1;
		_previousStripe[1] = 1;

		std::size_t iterCount = a.size() + b.size() - 1;
		for(std::size_t i = 0; i < iterCount; ++i) {

			// std::cout << std::endl;
			std::tie(currentStripeStart, currentStripeEnd) = getCurrentStripeRange(a.size(), b.size(), i, currentStripeStart, currentStripeEnd);
			bool lastTriangle = i >= b.size();
			if (!lastTriangle) {
				_currentStripe[0] = i + 2;
				_currentStripe[currentStripeEnd] = _currentStripe[0];
			}

			std::size_t xx = 0;
			// std::cout << "STRIPE " << i << " from " << currentStripeStart << " to " << currentStripeEnd << std::endl;
			// #pragma omp parallel for 
			for(std::size_t currentStripeIndex = currentStripeStart; currentStripeIndex < currentStripeEnd; ++currentStripeIndex) {
				C aChar = 0;
				++xx;
				if (lastTriangle) {
					aChar = a[a.size() - (xx)];
				} else {
					aChar = a[currentStripeEnd - currentStripeIndex - 1];
				}
				// std::cout << "Comparing "
				//  << (char)aChar << " and "
				//  << (char)b[currentStripeIndex - 1];


				DIST substitutionCost = (b[currentStripeIndex - 1] == aChar) ? 0 : 1;
				_currentStripe[currentStripeIndex] = std::min<DIST>({
					_previousStripe[currentStripeIndex] + (DIST)1, 
					_previousStripe[currentStripeIndex - 1] + (DIST)1,  
					_previousPreviousStripe[currentStripeIndex - 1] + substitutionCost 
				});
				// std::cout << " _currentStripe[currentStripeIndex] " << _currentStripe[currentStripeIndex] << std::endl;
			}

			// std::cout << "CURRENT:" << std::endl;
			// for(std::size_t i = currentStripeStart; i < currentStripeEnd; ++i) {
			// 	std::cout << _currentStripe[i] << " ";
			// }
			// std::cout << std::endl;

			// std::cout << "PREVIOUS:" << std::endl;
			// for(std::size_t i = currentStripeStart - 1; i < currentStripeEnd; ++i) {
			// 	std::cout << _previousStripe[i] << " ";
			// }
			// std::cout << std::endl;

			// std::cout << "PREVIOUS PREVIOUS:" << std::endl;
			// for(std::size_t i = currentStripeStart - 1; i < currentStripeEnd - 1; ++i) {
			// 	std::cout << _previousPreviousStripe[i] << " ";
			// }
			// std::cout << std::endl;
			// std::cout << std::endl;

			std::swap(_currentStripe, _previousPreviousStripe);
			std::swap(_previousStripe, _previousPreviousStripe);
		}
		return _previousStripe[currentStripeStart];

	}

	std::tuple<std::size_t, std::size_t> getCurrentStripeRange(std::size_t aLen, std::size_t bLen, std::size_t i, std::size_t currentStripeStart, std::size_t currentStripeEnd) {
		if (i < aLen) {
			return {currentStripeStart, ++currentStripeEnd};
		}

		if (i < bLen) {
			return {++currentStripeStart, ++currentStripeEnd};
		}

		return {++currentStripeStart, currentStripeEnd};
	}
};


#endif
