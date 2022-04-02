#ifndef LEVENSHTEIN_IMPLEMENTATION_HPP
#define LEVENSHTEIN_IMPLEMENTATION_HPP

#include <vector>
#include <algorithm>
#include <iostream>
#include <tuple>
#include <cassert>

template<typename C = char, typename DIST = std::size_t, bool DEBUG = false>
class EditDistance : public IEditDistance<C, DIST, DEBUG> {
public:

	/*
	 * \brief Perform the initialization of the functor (e.g., allocate memory buffers).
	 * \param len1, len2 Lengths of first and second string respectively.
	 */
	virtual void init(DIST len1, DIST len2) {
		_currentStripe.resize((std::size_t)std::max<DIST>(len1, len2) + 2);
		_previousStripe.resize((std::size_t)std::max<DIST>(len1, len2) + 2);
		_previousPreviousStripe.resize((std::size_t)std::max<DIST>(len1, len2) + 2);
	}

	/*
	 * \brief Compute the distance between two strings.
	 * \param str1, str2 Strings to be compared.
	 * \result The computed edit distance.
	 */
	virtual DIST compute(const std::vector<C> &str1, const std::vector<C> &str2) {
		if (str1.size() <= str2.size()) {
			return lev(str1, str2);
			// return lev3T(str1, str2);
		} else {
			return lev(str2, str1);
			// return lev3T(str2, str1);
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

			std::tie(currentStripeStart, currentStripeEnd) = getCurrentStripeRange(a.size(), b.size(), i, currentStripeStart, currentStripeEnd);
			bool lastTriangle = i >= b.size();
			_currentStripe[0] = i + 2;
			if (currentStripeEnd < _currentStripe.size()) {
				_currentStripe[currentStripeEnd] = _currentStripe[0];
			}

			#pragma omp parallel for
			for(std::size_t currentStripeIndex = currentStripeStart; currentStripeIndex < currentStripeEnd; ++currentStripeIndex) {
				C aChar = 0;
				if (lastTriangle) {
					aChar = a[a.size() - (currentStripeIndex - currentStripeStart + 1)];
				} else {
					aChar = a[currentStripeEnd - currentStripeIndex - 1];
				}
				DIST substitutionCost = (b[currentStripeIndex - 1] == aChar) ? 0 : 1;
				_currentStripe[currentStripeIndex] = std::min<DIST>({
					_previousStripe[currentStripeIndex] + (DIST)1, 
					_previousStripe[currentStripeIndex - 1] + (DIST)1,  
					_previousPreviousStripe[currentStripeIndex - 1] + substitutionCost 
				});
			}

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

DIST lev3T(const std::vector<C>& a, const std::vector<C>& b) { // a.len <= b.len
		std::size_t currentStripeStart = 1;
		std::size_t currentStripeEnd = 1;

		_previousStripe[0] = 1;
		_previousStripe[1] = 1;

		std::size_t iterCount = a.size() + b.size() - 1;
		std::size_t i = 0;
		for(; i < 1000; ++i) {

			std::tie(currentStripeStart, currentStripeEnd) = getCurrentStripeRange(a.size(), b.size(), i, currentStripeStart, currentStripeEnd);
			bool lastTriangle = i >= b.size();
			if (!lastTriangle) {
				_currentStripe[0] = i + 2;
				_currentStripe[currentStripeEnd] = _currentStripe[0];
			}

			for(std::size_t currentStripeIndex = currentStripeStart; currentStripeIndex < currentStripeEnd; ++currentStripeIndex) {
				C aChar = 0;
				if (lastTriangle) {
					aChar = a[a.size() - (currentStripeIndex - currentStripeStart + 1)];
				} else {
					aChar = a[currentStripeEnd - currentStripeIndex - 1];
				}

				DIST substitutionCost = (b[currentStripeIndex - 1] == aChar) ? 0 : 1;
				_currentStripe[currentStripeIndex] = std::min<DIST>({
					_previousStripe[currentStripeIndex] + (DIST)1, 
					_previousStripe[currentStripeIndex - 1] + (DIST)1,  
					_previousPreviousStripe[currentStripeIndex - 1] + substitutionCost 
				});
			}

			std::swap(_currentStripe, _previousPreviousStripe);
			std::swap(_previousStripe, _previousPreviousStripe);
		}

		for(; i < iterCount - 1000; ++i) {

			std::tie(currentStripeStart, currentStripeEnd) = getCurrentStripeRange(a.size(), b.size(), i, currentStripeStart, currentStripeEnd);
			bool lastTriangle = i >= b.size();
			if (!lastTriangle) {
				_currentStripe[0] = i + 2;
				_currentStripe[currentStripeEnd] = _currentStripe[0];
			}

			// #pragma omp parallel for 
			std::size_t blockSize = 256;
			std::size_t blockCount = (currentStripeEnd - currentStripeStart) / blockSize;
			for (std::size_t block = currentStripeStart; block < blockCount; block += blockSize) {

				for(std::size_t currentStripeIndex = block; currentStripeIndex < currentStripeEnd; ++currentStripeIndex) {
					C aChar = 0;
					if (lastTriangle) {
						aChar = a[a.size() - (currentStripeIndex - currentStripeStart + 1)];
					} else {
						aChar = a[currentStripeEnd - currentStripeIndex - 1];
					}

					DIST substitutionCost = (b[currentStripeIndex - 1] == aChar) ? 0 : 1;
					_currentStripe[currentStripeIndex] = std::min<DIST>({
						_previousStripe[currentStripeIndex] + (DIST)1, 
						_previousStripe[currentStripeIndex - 1] + (DIST)1,  
						_previousPreviousStripe[currentStripeIndex - 1] + substitutionCost 
					});
				}
			}

			std::swap(_currentStripe, _previousPreviousStripe);
			std::swap(_previousStripe, _previousPreviousStripe);
		}

		for(; i < iterCount; ++i) {

			std::tie(currentStripeStart, currentStripeEnd) = getCurrentStripeRange(a.size(), b.size(), i, currentStripeStart, currentStripeEnd);
			bool lastTriangle = i >= b.size();
			if (!lastTriangle) {
				_currentStripe[0] = i + 2;
				_currentStripe[currentStripeEnd] = _currentStripe[0];
			}

			for(std::size_t currentStripeIndex = currentStripeStart; currentStripeIndex < currentStripeEnd; ++currentStripeIndex) {
				C aChar = 0;
				if (lastTriangle) {
					aChar = a[a.size() - (currentStripeIndex - currentStripeStart + 1)];
				} else {
					aChar = a[currentStripeEnd - currentStripeIndex - 1];
				}

				DIST substitutionCost = (b[currentStripeIndex - 1] == aChar) ? 0 : 1;
				_currentStripe[currentStripeIndex] = std::min<DIST>({
					_previousStripe[currentStripeIndex] + (DIST)1, 
					_previousStripe[currentStripeIndex - 1] + (DIST)1,  
					_previousPreviousStripe[currentStripeIndex - 1] + substitutionCost 
				});
			}

			std::swap(_currentStripe, _previousPreviousStripe);
			std::swap(_previousStripe, _previousPreviousStripe);
		}
		return _previousStripe[currentStripeStart];
	}

};


#endif
