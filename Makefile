all:
	g++ -o makesievebase -lgmp -lgmpxx makesievebase.cc intpoly.cc mpz_poly.cc factorsmall.cc -std=c++11 -fopenmp -O3
	g++ -o slcsieve5dx -lgmp -lgmpxx slcsieve5dx.cc mpz_poly.cc L2lu64.cc factorsmall.cc intpoly.cc -std=c++11 -fopenmp -O3
	g++ -o slcsievehdxQ -lgmp -lgmpxx slcsievehdxQ.cc mpz_poly.cc L2lu64.cc L2lu128.cc factorsmall.cc intpoly.cc -lquadmath -fext-numeric-literals -std=c++11 -fopenmp -O3

debug:
	g++ -o makesievebase -lgmp -lgmpxx makesievebase.cc intpoly.cc mpz_poly.cc factorsmall.cc -std=c++11 -fopenmp -O0 -g
	g++ -o slcsieve5dx -lgmp -lgmpxx slcsieve5dx.cc mpz_poly.cc L2lu64.cc factorsmall.cc intpoly.cc -std=c++11 -fopenmp -O0 -g
	g++ -o slcsievehdxQ -lgmp -lgmpxx slcsievehdxQ.cc mpz_poly.cc L2lu64.cc L2lu128.cc factorsmall.cc intpoly.cc -lquadmath -fext-numeric-literals -std=c++11 -fopenmp -O0 -g

test:
	g++ -o fplll_test fplll_test.cc -lfplll -lgmp -lgmpxx
	g++ -o L2_test L2_test.cc L2lu64.cc -lgmp -lgmpxx

clean:
	rm makesievebase
	rm slcsieve5dx
	rm slcsievehdxQ
