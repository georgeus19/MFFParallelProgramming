#ifndef CUDA_POTENTIAL_IMPLEMENTATION_KERNELS_H
#define CUDA_POTENTIAL_IMPLEMENTATION_KERNELS_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <sstream>
#include <cstdint>

#include <data.hpp>
#include <algorithm>
#include <cassert>

/**
 * A stream exception that is base for all runtime errors.
 */
class CudaError : public std::exception
{
protected:
	std::string mMessage;	///< Internal buffer where the message is kept.
	cudaError_t mStatus;

public:
	CudaError(cudaError_t status = cudaSuccess) : std::exception(), mStatus(status) {}
	CudaError(const char *msg, cudaError_t status = cudaSuccess) : std::exception(), mMessage(msg), mStatus(status) {}
	CudaError(const std::string &msg, cudaError_t status = cudaSuccess) : std::exception(), mMessage(msg), mStatus(status) {}
	virtual ~CudaError() throw() {}

	virtual const char* what() const throw()
	{
		return mMessage.c_str();
	}

	// Overloading << operator that uses stringstream to append data to mMessage.
	template<typename T>
	CudaError& operator<<(const T &data)
	{
		std::stringstream stream;
		stream << mMessage << data;
		mMessage = stream.str();
		return *this;
	}
};


/**
 * CUDA error code check. This is internal function used by CUCH macro.
 */
inline void _cuda_check(cudaError_t status, int line, const char *srcFile, const char *errMsg = NULL)
{
	if (status != cudaSuccess) {
		throw (CudaError(status) << "CUDA Error (" << status << "): " << cudaGetErrorString(status) << "\n"
			<< "at " << srcFile << "[" << line << "]: " << errMsg);
	}
}

/**
 * Macro wrapper for CUDA calls checking.
 */
#define CUCH(status) _cuda_check(status, __LINE__, __FILE__, #status)

void run_add_repulsive_forces_kernel(Point<double>* points, ModelParameters<double>* mParams, std::uint32_t pointsSize, Point<double>* forces);

void run_add_compulsive_forces_kernel(std::uint32_t edgesSize, Point<double>* points, Edge<std::uint32_t>* edges, std::uint32_t* lengths,
									  ModelParameters<double>* mParams, Point<double>* forces);

// void run_add_compulsive_forces_kernel2(std::uint32_t pointsSize, Point<double>* points, Edge<std::uint32_t>* edges, std::uint32_t* lengths,
// 									  ModelParameters<double>* mParams, std::uint32_t** neighbourEdges, std::uint32_t* neighbourEdgesSizes, Point<double>* forces);

void run_apply_forces_kernel(std::uint32_t pointsSize, Point<double>* forces, ModelParameters<double>* mParams,
							 double fact, Point<double>* velocities, Point<double>* points);

#endif
