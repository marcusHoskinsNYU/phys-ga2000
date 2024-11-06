import numpy as np
import jax
import jax.numpy as jnp
import scipy.optimize as optimize
import csv
import matplotlib.pyplot as plt

def p(x, b0, b1):
    return 1 / (1 + np.exp(-(b0 + (b1*x))))

def negloglike(params, *args):
    b0 = params[0]
    b1 = params[1]
    ages = args[0]
    answers = args[1]

    length = len(ages)

    nll = 0
    for i in range(length):
        a = answers[i]
        x = ages[i]
        nll += (a*jnp.log(1 + jnp.exp(-(b0 + (b1*x))))) + ((1 - a)*jnp.log(1 + jnp.exp((b0 + (b1*x)))))

    return nll

def hessian(f):
    return jax.jacfwd(jax.grad(f))

negloglike_grad = jax.grad(negloglike)
h = hessian(negloglike)

"""  
Now we apply the data
"""

ages = []
answers = []

with open('survey.csv', mode = 'r') as survey:
    csvFile = csv.reader(survey)
    next(csvFile)
    for lines in csvFile:
        ages.append(float(lines[0]))
        answers.append(float(lines[1]))

ages = np.array(ages)
answers = np.array(answers)


pst = np.array([1., 1.])
r = optimize.minimize(negloglike, pst, jac=negloglike_grad,
                      args=(ages, answers), tol=1e-6)

print(r.x)

hmat = np.array(h(r.x, ages, answers))
covar = np.linalg.inv(hmat)
print(covar)
print(np.sqrt(np.diag(covar)))

minAge = np.min(ages)
maxAge = np.max(ages)

ctsAges = np.arange(minAge, maxAge, step=0.01)

plt.plot(ages, answers, 'k.', label='Data')
plt.plot(ctsAges, p(ctsAges, r.x[0], r.x[1]), 'r-', label='Logistic Model')
plt.xlabel('Ages')
plt.ylabel('Knowledge of Phrase')
plt.title('Survey Results vs Age')
plt.legend()
plt.show()