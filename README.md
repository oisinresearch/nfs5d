# Experimental 5d sieve for integer factorization, modified GNFS
This is a quick demo of a new idea which uses small integer linear combinations of
random number field elements to construct the sieving ideals.  It is largely
untested and totally unoptimised, but might be interesting.

### Overview
Hoping for people to experiment with this to see if we can get relations for e.g. RSA-155.
I do not claim this idea is any good yet, at least in its current experimental form.
But it appears to be a new way to "leverage" higher dimensional sieving for factoring.
It is interesting to use the additive structure of the number field.

Note that I have copied over several files from my PhD repository at
https://github.com/oisinresearch/latsieve, these are just useful subroutines.

At the moment, just try issuing

make

and you will get two files, makesievebase, which creates the factor base, and
slcsieve5dx, which is implements the 5d version of the idea.  I have included a polynomial
file for RSA-155.  Typical parameters are e.g.

./makesievebase rsa155a.poly 10000000 rsa155a.sievebase

./slcsieve5dx rsa155a.poly rsa155a.sievebase 4 100000 1000000000 5 100000 5000000 1000 3000 70 70 536870912 70 28

Since I only got this to compile about half an hour ago, all I have managed to get it to
do is show the divisibility actually does work (e.g. try getting the resultant of the
A*x + B with the RSA-155 polynomial, it will be divisible by primes in the factor base0),
but no relations for RSA-155 as of yet.

Of course this might change if the program parameters can be optimized.

### License
&copy; 2024, Oisin Robinson.
This work is published under the GNU General Public License (GPL) v3.0.
See the LICENSE file for complete statement.

