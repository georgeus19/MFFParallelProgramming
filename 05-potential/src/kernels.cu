#include "kernels.h"

__device__ double customAtomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

__global__ void add_repulsive_forces_kernel(Point<double>* points, ModelParameters<double>* mParams, std::uint32_t pointsSize, Point<double>* forces) {
	std::uint32_t p1 = blockIdx.x * blockDim.x + threadIdx.x;

	forces[p1].x = 0.0;
	forces[p1].y = 0.0;

	for (std::uint32_t p2 = 0; p2 < pointsSize; ++p2) {
		double dx = (double)points[p1].x - (double)points[p2].x;
		double dy = (double)points[p1].y - (double)points[p2].y;
		double sqLen = fmax(dx*dx + dy*dy, (double)0.0001);
		double factor = mParams->vertexRepulsion / (sqLen * (double)std::sqrt(sqLen));	// mul factor
		dx *= factor;
		dy *= factor;
		forces[p1].x += dx;
		forces[p1].y += dy;
	}
}

__global__ void add_compulsive_forces_kernel(Point<double>* points, Edge<std::uint32_t>* edges, std::uint32_t* lengths,
											 ModelParameters<double>* mParams, Point<double>* forces) {
	std::uint32_t edgeIdx = blockIdx.x * blockDim.x + threadIdx.x;

	std::uint32_t p1 = edges[edgeIdx].p1;
	std::uint32_t p2 = edges[edgeIdx].p2;
	std::uint32_t length = lengths[edgeIdx];

	double dx = (double)points[p2].x - (double)points[p1].x;
	double dy = (double)points[p2].y - (double)points[p1].y;
	double sqLen = dx*dx + dy*dy;
	double fact = (double)std::sqrt(sqLen) * mParams->edgeCompulsion / (double)(length);
	dx *= fact;
	dy *= fact;
	// forces[p1].x += dx;
	// forces[p1].y += dy;
	customAtomicAdd((double*)&(forces[p1].x), dx);
	customAtomicAdd((double*)&(forces[p1].y), dy);
	// forces[p2].x -= dx;
	// forces[p2].y -= dy;
	customAtomicAdd((double*)&(forces[p2].x), -dx);
	customAtomicAdd((double*)&(forces[p2].y), -dy);
}

__global__ void apply_forces_kernel(Point<double>* forces, ModelParameters<double>* mParams, double fact, Point<double>* velocities, Point<double>* points) {
	std::uint32_t pointIdx = blockIdx.x * blockDim.x + threadIdx.x;

	velocities[pointIdx].x = (velocities[pointIdx].x + (double)forces[pointIdx].x * fact) * mParams->slowdown;
	velocities[pointIdx].y = (velocities[pointIdx].y + (double)forces[pointIdx].y * fact) * mParams->slowdown;

	points[pointIdx].x += velocities[pointIdx].x * mParams->timeQuantum;
	points[pointIdx].y += velocities[pointIdx].y * mParams->timeQuantum;
}


void run_add_repulsive_forces_kernel(Point<double>* points, ModelParameters<double>* mParams, std::uint32_t pointsSize, Point<double>* forces) {
	add_repulsive_forces_kernel<<<pointsSize / 32, 32>>>(points, mParams, pointsSize, forces);
}

void run_add_compulsive_forces_kernel(std::uint32_t edgesSize, Point<double>* points, Edge<std::uint32_t>* edges, std::uint32_t* lengths,
									  ModelParameters<double>* mParams, Point<double>* forces) {
	add_compulsive_forces_kernel<<<edgesSize / 64, 64>>>(points, edges, lengths, mParams, forces);
}

void run_apply_forces_kernel(std::uint32_t pointsSize, Point<double>* forces, ModelParameters<double>* mParams,
							 double fact, Point<double>* velocities, Point<double>* points) {
	apply_forces_kernel<<<pointsSize / 64, 64>>>(forces, mParams, fact, velocities, points);
}
