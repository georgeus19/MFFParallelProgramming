#ifndef LEVENSHTEIN_IMPLEMENTATION_HPP
#define LEVENSHTEIN_IMPLEMENTATION_HPP

#include <vector>
#include <algorithm>
#include <iostream>
#include <tuple>
#include <cassert>
#include <omp.h>

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
		_availableCores = (std::size_t)omp_get_num_procs();

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
	std::size_t _availableCores;

	DIST lev(const std::vector<C>& a, const std::vector<C>& b) { // a.len <= b.len
		std::size_t currentStripeStart = 1;
		std::size_t currentStripeEnd = 1;

		_previousStripe[0] = 1;
		_previousStripe[1] = 1;

		std::size_t iterCount = a.size() + b.size() - 1;
		for(std::size_t i = 0; i < iterCount; ++i) {

			std::tie(currentStripeStart, currentStripeEnd) = getCurrentStripeRange(a.size(), b.size(), i, currentStripeStart, currentStripeEnd);
			bool lastTriangle = i >= b.size();
			_currentStripe[currentStripeStart - 1] = i + 2;
			if (currentStripeEnd < _currentStripe.size()) {
				_currentStripe[currentStripeEnd] = _currentStripe[currentStripeStart - 1];
			}

			std::size_t aOffset;

			if (lastTriangle) {
				aOffset = a.size() + currentStripeStart - 1;
			} else {
				aOffset = currentStripeEnd - 1;
			}

			std::size_t innerLoopIterCount = currentStripeEnd - currentStripeStart;
			std::size_t threadCount = std::min<std::size_t>(innerLoopIterCount / 1500, _availableCores);
			if (threadCount < 2) {
				for(std::size_t currentStripeIndex = currentStripeStart; currentStripeIndex < currentStripeEnd; ++currentStripeIndex) {
					oneLevStep(a, b, currentStripeIndex, aOffset);
				}
			} else {
				omp_set_num_threads(threadCount); 
				#pragma omp parallel for schedule(static)
				for(std::size_t currentStripeIndex = currentStripeStart; currentStripeIndex < currentStripeEnd; ++currentStripeIndex) {
					oneLevStep(a, b, currentStripeIndex, aOffset);
				}
			}

			std::swap(_currentStripe, _previousPreviousStripe);
			std::swap(_previousStripe, _previousPreviousStripe);
		}
		return _previousStripe[currentStripeStart];
	}

	void oneLevStep(const std::vector<C>& a, const std::vector<C>& b, std::size_t currentStripeIndex, std::size_t aOffset) {

		DIST substitutionCost = (DIST)(b[currentStripeIndex - 1] != a[aOffset - currentStripeIndex]);

		_currentStripe[currentStripeIndex] = std::min<DIST>(
			std::min<DIST>(_previousStripe[currentStripeIndex], _previousStripe[currentStripeIndex - 1]) + 1,
			_previousPreviousStripe[currentStripeIndex - 1] + substitutionCost
		);
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