#ifndef KMEANS_IMPLEMENTATION_HPP
#define KMEANS_IMPLEMENTATION_HPP

#include <functional>
#include "oneapi/tbb/parallel_for.h"
#include <iostream>
#include <cassert>
#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/parallel_reduce.h"
#include "oneapi/tbb/partitioner.h"
#include "oneapi/tbb/concurrent_vector.h"
#include "oneapi/tbb/task_group.h"
#include <sstream>
#include <thread>

template <typename POINT = point_t>
struct CentroidUpdate {
	POINT point;
	std::size_t count;

	CentroidUpdate() : point(), count(0) {
		point.x = 0;
		point.y = 0;
	}

	CentroidUpdate(std::int64_t x, std::int64_t y, std::size_t c) : point(), count(c) {
		point.x = x;
		point.y = y;
	}

	CentroidUpdate(const POINT& p, std::size_t c) : point(p), count(c) {}

	void update(const POINT& p) {
		point.x += p.x;
		point.y += p.y;
		count += 1;
	}

	void reset() {
		point.x = 0;
		point.y = 0;
		count = 0;
	}

};

template <typename T>
class Matrix {
private:
	std::vector<T> _data;
	std::size_t _rowCount;
	std::size_t _columnBuffer;
	std::size_t _columnCount;

	std::size_t get_index(std::size_t row, std::size_t col) {
		return row * _columnBuffer + col;
	}

public:

	Matrix() : _data(), _rowCount(0), _columnBuffer(0), _columnCount(0) {}

	T& at(std::size_t row, std::size_t col) {
		assert(row < _rowCount && col < _columnCount);
		assert(get_index(row, col) < _data.size());

		return _data[get_index(row, col)];
	}

	void resize(std::size_t rowCount, std::size_t columnBuffer, std::size_t columnCount) {
		_rowCount = rowCount;
		_columnBuffer = columnBuffer;
		_columnCount = columnCount;
		_data.resize(_rowCount * _columnBuffer);
	}

	size_t rowCount() {
		return _rowCount;
	}

	size_t columnCount() {
		return _columnCount;
	}
};

template <typename Range, typename POINT = point_t>
class ResetCentroidUpdatesBody{
private:
	Matrix<CentroidUpdate<POINT>>* _centroidUpdates;

public:

	ResetCentroidUpdatesBody(const ResetCentroidUpdatesBody& b)
		: _centroidUpdates(b._centroidUpdates) {}

	ResetCentroidUpdatesBody(Matrix<CentroidUpdate<POINT>>* centroidUpdates)
		: _centroidUpdates(centroidUpdates) {}

	~ResetCentroidUpdatesBody() {}

	void operator()(Range& range) const {
		for (auto taskIndex = range.begin(); taskIndex != range.end(); ++taskIndex) {
			for (std::size_t centroidIndex = 0; centroidIndex < _centroidUpdates->columnCount(); ++centroidIndex) {
				_centroidUpdates->at(taskIndex, centroidIndex).reset();
			}
		}
	}
};

struct NearestCentroidTask{
	const std::size_t taskId;
	const std::size_t from;
	const std::size_t to;

	NearestCentroidTask(std::size_t ti, std::size_t f, std::size_t t) : taskId(ti), from(f), to(t) {}
};


template<typename POINT = point_t, typename ASGN = std::uint8_t, bool DEBUG = false>
class KMeans : public IKMeans<POINT, ASGN, DEBUG> {
private:
	typedef typename POINT::coord_t coord_t;

	static constexpr size_t _pointsPerTask = 1024;
 
	Matrix<CentroidUpdate<POINT>> centroidUpdates;
	std::vector<std::vector<CentroidUpdate<POINT>>> _centroidUpdatesVector;

	static coord_t distance(const POINT &point, const POINT &centroid) {
		std::int64_t dx = (std::int64_t)point.x - (std::int64_t)centroid.x;
		std::int64_t dy = (std::int64_t)point.y - (std::int64_t)centroid.y;
		return (coord_t)(dx*dx + dy*dy);
	}

	static std::size_t getNearestCluster(const POINT &point, const std::vector<POINT> &centroids) {
		coord_t minDist = distance(point, centroids[0]);
		std::size_t nearest = 0;
		for (std::size_t i = 1; i < centroids.size(); ++i) {
			coord_t dist = distance(point, centroids[i]);
			if (dist < minDist) {
				minDist = dist;
				nearest = i;
			}
		}

		return nearest;
	}

	void computeNearestCentroids(
		const NearestCentroidTask& taskInfo,
		const std::vector<POINT>& points,
		const std::vector<POINT>& centroids,
		std::vector<ASGN>& assignments,
		bool lastIteration
	) {
		for (std::size_t columnIndex = 0; columnIndex < centroidUpdates.columnCount(); ++columnIndex) {
			centroidUpdates.at(taskInfo.taskId, columnIndex).reset();
			// _centroidUpdatesVector[rowIndex][columnIndex].reset();
		}
		
		for (std::size_t pointIndex = taskInfo.from; pointIndex != taskInfo.to; ++pointIndex) {
			std::size_t nearest_centroid = getNearestCluster(points[pointIndex], centroids);
			centroidUpdates.at(taskInfo.taskId, nearest_centroid).update(points[pointIndex]);
			// _centroidUpdatesVector[taskId][nearest_centroid].update(points[pointIndex]);

			if (lastIteration) {
				assignments[pointIndex] = (ASGN)nearest_centroid;
			}
		}
	}

public:
	/*
	 * \brief Perform the initialization of the functor (e.g., allocate memory buffers).
	 * \param points Number of points being clustered.
	 * \param k Number of clusters.
	 * \param iters Number of refining iterations.
	 */
	virtual void init(std::size_t points, std::size_t k, std::size_t iters) {
		centroidUpdates.resize((points / _pointsPerTask) + 1, 256, k);
		_centroidUpdatesVector.resize((points / _pointsPerTask) + 1);
		for(auto&& v : _centroidUpdatesVector) {
			v.resize(k);
		}
	}

	/*
	 * \brief Perform the clustering and return the cluster centroids and point assignment
	 *		yielded by the last iteration.
	 * \note First k points are taken as initial centroids for first iteration.
	 * \param points Vector with input points.
	 * \param k Number of clusters.
	 * \param iters Number of refining iterations.
	 * \param centroids Vector where the final cluster centroids should be stored.
	 * \param assignments Vector where the final assignment of the points should be stored.
	 *		The indices should correspond to point indices in 'points' vector.
	 */
	virtual void compute(const std::vector<POINT> &points, std::size_t k, std::size_t iters,
		std::vector<POINT> &centroids, std::vector<ASGN> &assignments) {
		
		// Prepare for the first iteration
		centroids.resize(k);
		assignments.resize(points.size());
		for (std::size_t i = 0; i < k; ++i)
			centroids[i] = points[i];

		// Run the k-means refinements
		while (iters > 0) {
			--iters;
			bool lastIteration = iters == 0;

			// if (iters == 0) {
			// 	for (std::size_t i = 0; i < points.size(); ++i) {
			// 		std::size_t nearest = getNearestCluster(Tpoints[i], centroids);
			// 		assignments[i] = (ASGN)nearest;
			// 	}
			// }

			// for (std::size_t rowIndex = 0; rowIndex < centroidUpdates.rowCount(); ++rowIndex) {
			// 	for (std::size_t columnIndex = 0; columnIndex < centroidUpdates.columnCount(); ++columnIndex) {
			// 		centroidUpdates.at(rowIndex, columnIndex).reset();
			// 		// _centroidUpdatesVector[rowIndex][columnIndex].reset();
			// 	}
			// }

			// ResetCentroidUpdatesBody<oneapi::tbb::blocked_range<std::size_t>> resetCentroidUpdates{&centroidUpdates};
			// const std::size_t rowsPerTask = 1024;
			// size_t resetRangeEnd = 0;
			// for (std::size_t rowIndex = 0; rowIndex < centroidUpdates.rowCount(); rowIndex += rowsPerTask) {
			// 	resetRangeEnd = rowIndex + rowsPerTask;
			// 	if (resetRangeEnd > centroidUpdates.rowCount()) {
			// 		resetRangeEnd = centroidUpdates.rowCount();
			// 	}

			// 	oneapi::tbb::blocked_range<std::size_t> range{rowIndex, resetRangeEnd, rowsPerTask};
			// 	resetCentroidUpdates(range);

			// }

			// oneapi::tbb::blocked_range<std::size_t> resetRange{0, centroidUpdates.rowCount(), rowsPerTask};
			// oneapi::tbb::affinity_partitioner partitioner{}; 
			// oneapi::tbb::parallel_for(resetRange, resetCentroidUpdates, partitioner);
			

			oneapi::tbb::task_group nearestCentroidTasks{};

			std::size_t taskId = 0;
			for (std::size_t rangeStart = 0; rangeStart < points.size(); rangeStart += _pointsPerTask) {
				std::size_t rangeEnd = rangeStart + _pointsPerTask;
				if (rangeEnd > points.size()) {
					rangeEnd = points.size();
				}
				nearestCentroidTasks.run([=, &points, &centroids, &assignments](){
					computeNearestCentroids(NearestCentroidTask{taskId, rangeStart, rangeEnd}, points, centroids, assignments, lastIteration);
					
				});
				++taskId;

			}
			nearestCentroidTasks.wait();


			oneapi::tbb::parallel_for<std::size_t>((std::size_t)0, centroidUpdates.columnCount(), [&](std::size_t centroidIndex) {
				std::int64_t x = 0;
				std::int64_t y = 0;
				std::int64_t count = 0;
				for(std::size_t taskIndex = 0; taskIndex < centroidUpdates.rowCount(); ++taskIndex) {
					x += centroidUpdates.at(taskIndex, centroidIndex).point.x;
					y += centroidUpdates.at(taskIndex, centroidIndex).point.y;
					count += centroidUpdates.at(taskIndex, centroidIndex).count;

					// x += _centroidUpdatesVector[taskIndex][centroidIndex].point.x;
					// y += _centroidUpdatesVector[taskIndex][centroidIndex].point.y;
					// count += _centroidUpdatesVector[taskIndex][centroidIndex].count;

				}

				if (count > 0) {
					centroids[centroidIndex].x = x / (std::int64_t)count;
					centroids[centroidIndex].y = y / (std::int64_t)count;

				}
			});

			// for(std::size_t centroidIndex = 0; centroidIndex < centroidUpdates.columnCount(); ++centroidIndex) {
				
			// 	std::int64_t x = 0;
			// 	std::int64_t y = 0;
			// 	std::int64_t count = 0;
			// 	for(std::size_t taskIndex = 0; taskIndex < centroidUpdates.rowCount(); ++taskIndex) {
			// 		x += centroidUpdates.at(taskIndex, centroidIndex).point.x;
			// 		y += centroidUpdates.at(taskIndex, centroidIndex).point.y;
			// 		count += centroidUpdates.at(taskIndex, centroidIndex).count;

			// 		// x += _centroidUpdatesVector[taskIndex][centroidIndex].point.x;
			// 		// y += _centroidUpdatesVector[taskIndex][centroidIndex].point.y;
			// 		// count += _centroidUpdatesVector[taskIndex][centroidIndex].count;

			// 	}

			// 	if (count > 0) {
			// 		centroids[centroidIndex].x = x / (std::int64_t)count;
			// 		centroids[centroidIndex].y = y / (std::int64_t)count;

			// 	}
			// }
		


		}
	}
};


#endif
