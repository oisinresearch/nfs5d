all:
	g++ -o slcsievehdavx512 slcsievehdavx512.cc mpz_poly.cc L2lu64.cc factorsmall.cc intpoly.cc -lquadmath -fext-numeric-literals -std=c++11 -fopenmp -O3 -lgmp -lgmpxx -mavx512f -mavx512vl -mavx512bw
	g++ -o makesievebase makesievebase.cc intpoly.cc mpz_poly.cc factorsmall.cc -std=c++11 -fopenmp -O3 -lgmp -lgmpxx
	g++ -o makesievebasemono makesievebasemono.cc intpoly.cc mpz_poly.cc factorsmall.cc -std=c++11 -fopenmp -O3 -lgmp -lgmpxx
	g++ -o slcsieve5dx slcsieve5dx.cc mpz_poly.cc L2lu64.cc factorsmall.cc intpoly.cc -std=c++11 -fopenmp -O3 -lgmp -lgmpxx
	g++ -o slcsievehdx slcsievehdx.cc mpz_poly.cc L2lu64.cc factorsmall.cc intpoly.cc -lquadmath -fext-numeric-literals -std=c++11 -fopenmp -O3 -lgmp -lgmpxx
	g++ -o slcsievehdx_fplll slcsievehdx_fplll.cc mpz_poly.cc factorsmall.cc intpoly.cc -fext-numeric-literals -std=c++11 -fopenmp -O3 -lgmp -lgmpxx -lfplll
	g++ -o slcsievehdxmono slcsievehdxmono.cc L2lu64.cc mpz_poly.cc factorsmall.cc intpoly.cc -fext-numeric-literals -std=c++11 -fopenmp -O3 -lgmp -lgmpxx -lfplll
	g++ -o slcsievehdxQ slcsievehdxQ.cc mpz_poly.cc L2lu64.cc L2lu128.cc factorsmall.cc intpoly.cc -lquadmath -fext-numeric-literals -std=c++11 -fopenmp -O3 -lgmp -lgmpxx
	g++ -o slcsievehdxQ_fplll slcsievehdxQ_fplll.cc mpz_poly.cc L2lu64.cc factorsmall.cc intpoly.cc -lquadmath -fext-numeric-literals -std=c++11 -fopenmp -O3 -lgmp -lgmpxx -lfplll
	g++ -o slcsievehdxQ2_fplll slcsievehdxQ2_fplll.cc mpz_poly.cc L2lu64.cc factorsmall.cc intpoly.cc -lquadmath -fext-numeric-literals -std=c++11 -fopenmp -O3 -lgmp -lgmpxx -lfplll
	g++ -o showlattice showlattice.cc mpz_poly.cc L2lu64.cc factorsmall.cc intpoly.cc -lquadmath -fext-numeric-literals -std=c++11 -fopenmp -O3 -lgmp -lgmpxx -lfplll

debug:
	g++ -o slcsievehdavx512 slcsievehdavx512.cc mpz_poly.cc L2lu64.cc factorsmall.cc intpoly.cc -lquadmath -fext-numeric-literals -std=c++11 -fopenmp -O0 -g -lgmp -lgmpxx -mavx512f -mavx512vl -mavx512bw
	g++ -o makesievebase makesievebase.cc intpoly.cc mpz_poly.cc factorsmall.cc -std=c++11 -fopenmp -O0 -g -lgmp -lgmpxx
	g++ -o makesievebasemono makesievebasemono.cc intpoly.cc mpz_poly.cc factorsmall.cc -std=c++11 -fopenmp -O0 -g -lgmp -lgmpxx
	g++ -o slcsieve5dx slcsieve5dx.cc mpz_poly.cc L2lu64.cc factorsmall.cc intpoly.cc -std=c++11 -fopenmp -O0 -g -lgmp -lgmpxx
	g++ -o slcsievehdx slcsievehdx.cc mpz_poly.cc L2lu64.cc factorsmall.cc intpoly.cc -lquadmath -fext-numeric-literals -std=c++11 -fopenmp -O0 -g -lgmp -lgmpxx
	g++ -o slcsievehdx_fplll slcsievehdx_fplll.cc mpz_poly.cc factorsmall.cc intpoly.cc -fext-numeric-literals -std=c++11 -fopenmp -O0 -g -lgmp -lgmpxx -lfplll
	g++ -o slcsievehdxmono slcsievehdxmono.cc L2lu64.cc mpz_poly.cc factorsmall.cc intpoly.cc -fext-numeric-literals -std=c++11 -fopenmp -O0 -g -lgmp -lgmpxx -lfplll
	g++ -o slcsievehdxQ slcsievehdxQ.cc mpz_poly.cc L2lu64.cc L2lu128.cc factorsmall.cc intpoly.cc -lquadmath -fext-numeric-literals -std=c++11 -fopenmp -O0 -g -lgmp -lgmpxx
	g++ -o slcsievehdxQ_fplll slcsievehdxQ_fplll.cc mpz_poly.cc L2lu64.cc factorsmall.cc intpoly.cc -lquadmath -fext-numeric-literals -std=c++11 -fopenmp -O0 -g -lgmp -lgmpxx -lfplll
	g++ -o slcsievehdxQ2_fplll slcsievehdxQ2_fplll.cc mpz_poly.cc L2lu64.cc factorsmall.cc intpoly.cc -lquadmath -fext-numeric-literals -std=c++11 -fopenmp -O0 -g -lgmp -lgmpxx -lfplll
	g++ -o showlattice showlattice.cc mpz_poly.cc L2lu64.cc factorsmall.cc intpoly.cc -lquadmath -fext-numeric-literals -std=c++11 -fopenmp -O0 -g -lgmp -lgmpxx -lfplll

test:
	g++ -o fplll_test fplll_test.cc -lgmp -lgmpxx -lfplll
	g++ -o L2_test L2_test.cc L2lu64.cc -lgmp -lgmpxx

clean:
	rm -f slcsievehdavx512
	rm -f makesievebase
	rm -f makesievebasemono
	rm -f slcsieve5dx
	rm -f slcsievehdx
	rm -f slcsievehdx_fplll
	rm -f slcsievehdxmono
	rm -f slcsievehdxQ
	rm -f slcsievehdxQ_fplll
	rm -f slcsievehdxQ2_fplll
	rm -f fplll_test
	rm -f L2_test
	rm -f showlattice
