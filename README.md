<h1>Thesis Documentation</h1>
<h3>Code Overview</h3>

* For the full algorithm for polynomial decomposition, run <b><u>train10.py</u></b>. All the previous training files are
  development
  versions that led to the product that is presented in train.10.py.

* <b><u>train11.py</u></b> check the decomposition of polynomial that came from a composition of two polynomials. It is
  a test
  file that checks the decomposition of polynomials, with different degrees than those of the composed polynomials.

* <b><u>train12.py</u></b> check the decomposition for extremely large polynomials. Its seems like it gets pretty close.
  I believe it is the aggressive regularization on the largest coefficient that enforces it.

* <b><u>train13.py</u></b> tries to decompose random polynomials with random coefficients.

* <b><u>train14.py</u></b> Address the disadvantages of train12.py by forcing rounding and sophisticated loss adjustments.

* <b><u>rank_directions.py</u></b> Round the values of Q in all possible combinations (up or down) and rank all the different possibilities by their loss for the same P as the model outputs.
* <b><u>find_closest_solution.py</u></b> Find a composition to the problem which its coefficients are the closest to those of the polynomials the algorithm outputted.
* <b><u>train15.py</u></b> Tries to address the extreme behavior of P's coefficients by applying l2 regularization for them.
* <b><u>train16.py</u></b> Omits the l2 regularization for P's coefficients. By that the algorithm converges much better.
* <b><u>train17.py</u></b> Is a HP tuned version of train16.py.
* <b><u>train18.py</u></b> Normalizes the coefficients of R such that the coefficient of the highest degree is 1.
* <b><u>train19.py</u></b> Try again to decompose specific polynomials.