<h1>Thesis Documentation</h1>
<h3>Code Overview</h3>

* For the full algorithm for polynomial decomposition, run <b><u>train10.py</u></b>. All the previous training files are
  development
  versions that led to the product that is presented in train.10.py.

* ```train11.py``` check the decomposition of polynomial that came from a composition of two polynomials. It is
  a test
  file that checks the decomposition of polynomials, with different degrees than those of the composed polynomials.

* ```train12.py``` check the decomposition for extremely large polynomials. Its seems like it gets pretty close.
  I believe it is the aggressive regularization on the largest coefficient that enforces it.

* ```train13.py``` tries to decompose random polynomials with random coefficients.

* ```train14.py``` Address the disadvantages of train12.py by forcing rounding and sophisticated loss adjustments.

* ```rank_directions.py``` Round the values of Q in all possible combinations (up or down) and rank all the different possibilities by their loss for the same P as the model outputs.
* ```find_closest_solution.py``` Find a composition to the problem which its coefficients are the closest to those of the polynomials the algorithm outputted.
* ```train15.py``` Tries to address the extreme behavior of P's coefficients by applying l2 regularization for them.
* ```train16.py``` Omits the l2 regularization for P's coefficients. By that the algorithm converges much better.
* ```train17.py``` Is a HP tuned version of train16.py.
* ```train18.py``` Normalizes the coefficients of R such that the coefficient of the highest degree is 1.
* ```train19.py``` Try again to decompose specific polynomials.

<h3>Experiments</h3>

In this section we start expiermenting with the algorithm and try to decompose polynomials that are composed of two polynomials.
We first build a dataset using ```create_dataset.py``` with vary degrees and scales of coefficients.

Then, we applied the valina generic algorithm on the dataset, as is, using ```pipeline.py```.

Points to note:
* There are modifications to the algorithms that need to be done in order to adjust for different scales and varying degrees.
* We need to better understand the cases where the algorithm fails.

Right now, I'm trying to figure out how can I make train18 more tolerant for cases which could converge.