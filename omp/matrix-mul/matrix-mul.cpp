#include "matrix.hpp"

#include "omp.h"
#include <iostream>

template<class M>
void serial_matrix_mul(const M &m1, const M &m2, M &res)
{
	if (m1.m() != m2.n()) throw std::runtime_error("Incompatible proportions of matrices.");

	for (std::size_t i = 0; i < m1.n(); ++i)
		for (std::size_t j = 0; j < m2.m(); ++j)
		{
			typename M::value_t sum = (typename M::value_t)0;
			for (std::size_t k = 0; k < m1.m(); ++k)
				sum += m1.at(i, k) * m2.at(k, j);
			res.at(i, j) = sum;
		}
}

template<class M>
void para_matrix_mul(const M& m1, const M& m2, M& res)
{
	if(m1.m()!=m2.n()) throw std::runtime_error("Incompatible proportions of matrices.");

	int i = 0;
	int j = 0;
	int k = 0;
	const int m1n = m1.n();
	const int m2m = m2.m();
	const int m1m = m1.m();
	#pragma omp parallel for collapse(2) private(i) private(j) private(k) schedule(static)
	for(i = 0; i<m1n; ++i)
		for(j = 0; j<m2m; ++j)
		{
			typename M::value_t sum = (typename M::value_t)0;
			for(k = 0; k<m1m; ++k)
				sum += m1.at(i, k)*m2.at(k, j);
			res.at(i, j) = sum;
		}
}


int main(int argc, char *argv[])
{
	typedef float value_t;

	const std::size_t N = 1024;
	std::cout << "Preparing input matrices " << N << "x" << N << " ..." << std::endl;
	Matrix<value_t> m1(N, N), m2(N, N);
	matrix_fill_random(m1);
	matrix_fill_random(m2);

	double tstart, tend;

	std::cout << "Serial multiplication ... ";
	std::cout.flush();
	Matrix<value_t> sres(N, N);

	tstart = omp_get_wtime();
	serial_matrix_mul(m1, m2, sres);
	tend = omp_get_wtime();
	std::cout << (tend - tstart) << "s" << std::endl;

	std::cout<<"Parallel multiplication ... ";
	std::cout.flush();
	Matrix<value_t> pres(N, N);

	tstart = omp_get_wtime();
	para_matrix_mul(m1, m2, pres);
	tend = omp_get_wtime();
	std::cout<<(tend-tstart)<<"s"<<std::endl;

	return 0;
}
