#include "matrix.hpp"

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/tick_count.h>
#include <tbb/task.h>

#include <random>
#include <vector>
#include <iostream>


/**
 * Naive implementation of serial transposition.
 */
template<typename T, class O>
void serial_transpose(Matrix<T, O> &matrix)
{
	for (std::size_t i = 0; i < matrix.m(); ++i)
		for (std::size_t j = i + 1; j < matrix.n(); ++j)
			std::swap(matrix.at(i, j), matrix.at(j, i));
}


/**
 * Verify that two matrices hold the same values.
 */
template<typename T, class O>
bool verify(const Matrix<T, O> &m1, const Matrix<T, O> &m2)
{
	if (m1.m() != m2.m() || m1.n() != m1.n()) {
		std::cerr << "The verified matrices differ in size (" << m1.m() << "x" << m1.n() << " vs " << m2.m() << "x" << m2.n() << ")." << std::endl;
		return false;
	}
	
	std::size_t errors = 0;
	for (std::size_t i = 0; i < m1.m(); ++i)
		for (std::size_t j = i + 1; j < m2.n(); ++j) {
			if (m1.at(i, j) != m2.at(i, j)) {
				if (errors == 0) std::cerr << "FAILED" << std::endl;
				if (++errors <= 20) {	// preview first 20 errors
					std::cerr << "Mismatch at [" << i << "," << j << "]: " << m1.at(i, j) << " != " << m2.at(i, j) << std::endl;
				}
			}
		}

	if (errors) {
		std::cerr << "Total " << errors << " errors found." << std::endl;
		return false;
	}
	return true;
}



/*
 * Parallel implementations ...
 */

// Naive parallel solution ... trivial parallel for over top-level
template<typename T, class O>
void parallel_transpose_naive_for(Matrix<T, O> &matrix)
{
	tbb::parallel_for(size_t(0), matrix.m(), [&](size_t i) {
		for (std::size_t j = i + 1; j < matrix.n(); ++j)
			std::swap(matrix.at(i, j), matrix.at(j, i));
	});
}


template<typename T, class O>
void serial_transpose_divide_conquer(Matrix<T, O> &matrix, std::size_t fromI, std::size_t fromJ, std::size_t size)
{
	if (size >= 8) {
		std::size_t half = size / 2;
		serial_transpose_divide_conquer(matrix, fromI, fromJ, half);
		serial_transpose_divide_conquer(matrix, fromI + half, fromJ + half, half);
		serial_transpose_divide_conquer(matrix, fromI + half, fromJ, half);
		serial_transpose_divide_conquer(matrix, fromI, fromJ + half, half);


		std::size_t toI = fromI + half;
		std::size_t toJ = fromJ + half;
		for (std::size_t i = fromI; i < toI; ++i)
			for (std::size_t j = fromJ; j < toJ; ++j)
				std::swap(matrix.at(i+half, j), matrix.at(i, j+half));

	}
	else {
		for (std::size_t i = 0; i < size; ++i)
			for (std::size_t j = i + 1; j < size; ++j)
				std::swap(matrix.at(i + fromI, j + fromJ), matrix.at(j + fromI, i + fromJ));
	}
}


/**
 * Swaps two tiles of the matrix on positions i+size,j a and i,j+size.
 */
template<typename T, class O>
class SwapTilesTask : public tbb::task
{
private:
	Matrix<T, O> &mMatrix;
	std::size_t mI, mJ, mSize;

public:
	SwapTilesTask(Matrix<T, O> &matrix, std::size_t i, std::size_t j, std::size_t size)
		: mMatrix(matrix), mI(i), mJ(j), mSize(size) {}
	
	virtual tbb::task* execute()
	{
		tbb::parallel_for(size_t(0), mSize, [&](size_t i) {
			for (std::size_t j = 0; j < mSize; ++j) {
				std::swap(mMatrix.at(mI + i + mSize, mJ + j), mMatrix.at(mI + i, mJ + j + mSize));
			}
		});
		return nullptr;
	}
};


/**
 * Perform the transosition recursively.
 */
template<typename T, class O>
class TransposeTask : public tbb::task
{
private:
	Matrix<T, O> &mMatrix;
	std::size_t mI, mJ, mSize;

public:
	TransposeTask(Matrix<T, O> &matrix, std::size_t i, std::size_t j, std::size_t size)
		: mMatrix(matrix), mI(i), mJ(j), mSize(size) {}

	virtual tbb::task* execute()
	{
		if (mSize <= 64) {
			serial_transpose_divide_conquer(mMatrix, mI, mJ, mSize);
			return nullptr;
		}

		mSize /= 2;
		SwapTilesTask<T, O> &cont = *new(this->allocate_continuation()) SwapTilesTask<T, O>(mMatrix, mI, mJ, mSize);
		tbb::task_list children;
		children.push_back(*new(cont.allocate_child()) TransposeTask<T, O>(mMatrix, mI, mJ + mSize, mSize));
		children.push_back(*new(cont.allocate_child()) TransposeTask<T, O>(mMatrix, mI + mSize, mJ, mSize));
		children.push_back(*new(cont.allocate_child()) TransposeTask<T, O>(mMatrix, mI + mSize, mJ + mSize, mSize));
		this->recycle_as_child_of(cont);
		cont.set_ref_count(4);			// continuation will start, after his 4 children complete
		tbb::task::spawn(children);	
		return this;					// this is actually essential, otherwise the recycled this will never be spawned
	}
};



template<typename T, class O>
void parallel_transpose_divide_conquer(Matrix<T, O> &matrix)
{
	TransposeTask<T,O> &root = *new(tbb::task::allocate_root()) TransposeTask<T, O>(matrix, 0, 0, matrix.m());
	tbb::task::spawn_root_and_wait(root);
}


/**
 * Parallel matrix transposition using TBB
 */
template<typename T, class O>
void parallel_transpose(Matrix<T, O> &matrix)
{
	//parallel_transpose_naive_for(matrix);
	//serial_transpose_divide_conquer(matrix, 0, 0, matrix.m());
	parallel_transpose_divide_conquer(matrix);
}



int main(int argc, char *argv[])
{
	std::size_t n = 16*1024;
	if ((n | (n - 1)) != (2 * n - 1))
		std::cerr << "Warining: n (" << n << ") is not a power of 2." << std::endl;

	// Prepare data ...
	Matrix<std::uint64_t> matrix(n, n), matrixTmp(n, n), matrixVerif(n, n);
	matrix_fill_seq(matrix);

	// Run and time serial version algorithms ...
	matrixTmp.copyFrom(matrix);
	matrixTmp.touch();
	tbb::tick_count tstart1 = tbb::tick_count::now();
	serial_transpose(matrixTmp);
	tbb::tick_count tend1 = tbb::tick_count::now();
	matrixVerif.copyFrom(matrixTmp);

	// Run and time parallel version ...
	matrixTmp.copyFrom(matrix);
	matrixTmp.touch();
	tbb::tick_count tstart2 = tbb::tick_count::now();
	parallel_transpose(matrixTmp);
	tbb::tick_count tend2 = tbb::tick_count::now();

	// Print times ...
	std::cout << "serial time: " << (tend1 - tstart1).seconds() << "s" << std::endl;
	std::cout << "tbb time :   " << (tend2 - tstart2).seconds() << "s" << std::endl;

	// Verify results ...
	std::cout << "Verifying ... ";
	std::cout.flush();
	if (verify(matrixTmp, matrixVerif)) {
		std::cout << "OK" << std::endl;
		return 0;
	}
	else
		return 1;
}
