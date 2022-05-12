#ifndef CUDA_POTENTIAL_IMPLEMENTATION_HPP
#define CUDA_POTENTIAL_IMPLEMENTATION_HPP

#include "kernels.h"

#include <interface.hpp>
#include <data.hpp>

#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

/*
 * Final implementation of the tested program.
 */
template<typename F = float, typename IDX_T = std::uint32_t, typename LEN_T = std::uint32_t>
class ProgramPotential : public IProgramPotential<F, IDX_T, LEN_T> {
public:
	typedef F coord_t;		// Type of point coordinates.
	typedef coord_t real_t;	// Type of additional float parameters.
	typedef IDX_T index_t;
	typedef LEN_T length_t;
	typedef Point<coord_t> point_t;
	typedef Edge<index_t> edge_t;

	using IProgramPotential<F, IDX_T, LEN_T>::mParams;

private:

	edge_t* _cuEdges = nullptr;
	point_t* _cuPoints = nullptr;
	length_t* _cuLengths = nullptr;
	point_t* _cuVelocities = nullptr;
	point_t* _cuForces = nullptr;

	ModelParameters<real_t>* _cuModelParams = nullptr;

	index_t _pointsSize;
	index_t _edgesSize;

	bool _sendPointsToGpu = true;

public:
	virtual void initialize(index_t points, const std::vector<edge_t>& edges, const std::vector<length_t>& lengths, index_t iterations) {
		CUCH(cudaSetDevice(0));

		CUCH(cudaMalloc((void**)&_cuPoints, points * sizeof(point_t)));

		CUCH(cudaMalloc((void**)&_cuEdges, edges.size() * sizeof(edge_t)));
		CUCH(cudaMemcpy(_cuEdges, edges.data(), edges.size() * sizeof(edge_t), cudaMemcpyHostToDevice));

		CUCH(cudaMalloc((void**)&_cuLengths, lengths.size() * sizeof(length_t)));
		CUCH(cudaMemcpy(_cuLengths, lengths.data(), lengths.size() * sizeof(length_t), cudaMemcpyHostToDevice));
		
		std::vector<point_t> forcesInitialValues(points, point_t{0.0, 0.0});
		CUCH(cudaMalloc((void**)&_cuForces, points * sizeof(point_t)));
		CUCH(cudaMemcpy(_cuForces, forcesInitialValues.data(), points * sizeof(point_t), cudaMemcpyHostToDevice));

		std::vector<point_t> velocitiesInitialValues(points, point_t{0.0, 0.0});
		CUCH(cudaMalloc((void**)&_cuVelocities, points * sizeof(point_t)));
		CUCH(cudaMemcpy(_cuVelocities, velocitiesInitialValues.data(), points * sizeof(point_t), cudaMemcpyHostToDevice));

		CUCH(cudaMalloc((void**)&_cuModelParams, sizeof(ModelParameters<real_t>)));
		CUCH(cudaMemcpy(_cuModelParams, &mParams, sizeof(ModelParameters<real_t>), cudaMemcpyHostToDevice));

		CUCH(cudaMalloc((void**)&_cuModelParams, sizeof(ModelParameters<real_t>)));
		CUCH(cudaMemcpy(_cuModelParams, &mParams, sizeof(ModelParameters<real_t>), cudaMemcpyHostToDevice));

		_pointsSize = points;
		_edgesSize = edges.size();

		CUCH(cudaDeviceSynchronize());

	}

	virtual void iteration(std::vector<point_t>& points) {
		/*
		 * Perform one iteration of the simulation and update positions of the points.
		 */
		if (_sendPointsToGpu) {
			CUCH(cudaMemcpy(_cuPoints, points.data(), points.size() * sizeof(point_t), cudaMemcpyHostToDevice));
			_sendPointsToGpu = false;
		}

		run_add_repulsive_forces_kernel(_cuPoints, _cuModelParams, _pointsSize, _cuForces);
		CUCH(cudaGetLastError());

		run_add_compulsive_forces_kernel(_edgesSize, _cuPoints, _cuEdges, _cuLengths, _cuModelParams, _cuForces);
		CUCH(cudaGetLastError());

		real_t fact = mParams.timeQuantum / mParams.vertexMass;
		run_apply_forces_kernel(_pointsSize, _cuForces, _cuModelParams, fact, _cuVelocities, _cuPoints);
		CUCH(cudaGetLastError());

		CUCH(cudaDeviceSynchronize());
		CUCH(cudaMemcpy(points.data(), _cuPoints, points.size() * sizeof(point_t), cudaMemcpyDeviceToHost));
	}


	virtual void getVelocities(std::vector<point_t>& velocities) {
		/*
		 * Retrieve the velocities buffer from the GPU.
		 * This operation is for vreification only and it does not have to be efficient.
		 */
		velocities.resize(_pointsSize);
		CUCH(cudaMemcpy(velocities.data(), _cuVelocities, velocities.size() * sizeof(point_t), cudaMemcpyDeviceToHost));
	}
};


#endif
