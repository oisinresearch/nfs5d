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

export OMP_NUM_THREADS=16
./makesievebase rsa155b.poly 10000000 rsa155b.sievebase

./slcsieve5dx rsa155b.poly rsa155b.sievebase 4 1000 10000000 20 50000 10000000 60 100 55 50 536870912 11 28

After changing the divisibility strategy with a better lattice reduction,
the program is now producing valid relations (try the above command line on
the current main branch).  There is still of course much room for improvement,
there has been no real performance optimization so far.

### License
&copy; 2024, Oisin Robinson.
This work is published under the GNU General Public License (GPL) v3.0.
See the LICENSE file for complete statement.

