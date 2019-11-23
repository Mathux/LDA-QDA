import numpy as np
import math
from matplotlib import pyplot
from scipy.io import loadmat

# fonction pour lire un fichier
def loadfile(strfile):
    fileo = open(strfile, "r") 
    y = []
    for line in fileo: 
        y.append(list(map(float, line.split())))
    fileo.close()
    return np.array(y)
    
# load trains
trainA = loadfile("classificationA.train")
trainB = loadfile("classificationB.train")
trainC = loadfile("classificationC.train")

# load tests
testA = loadfile("classificationA.test")
testB = loadfile("classificationB.test")
testC = loadfile("classificationC.test")

# fonction pour l'affichage des points
def show_bicolor(tab, name, show=True):
    t0x = [x[0] for x in tab if x[2]==0]
    t0y = [x[1] for x in tab if x[2]==0]
    t1x = [x[0] for x in tab if x[2]==1]
    t1y = [x[1] for x in tab if x[2]==1]
    pyplot.scatter(t0x, t0y, c = 'red', marker = '.')
    pyplot.scatter(t1x, t1y, c = 'green', marker = '.')
    #pyplot.legend()
    pyplot.title(name)
    if show:
        pyplot.show()

show_bicolor(trainA, 'Ensembles d\'apprentissage A')
show_bicolor(trainB, 'Ensembles d\'apprentissage B')
show_bicolor(trainC, 'Ensembles d\'apprentissage C')

def classifieur_reglin(tab):
    Y = tab.T[2]
    X = np.array([tab[:,0],tab[:,1], np.ones(tab[:,0].shape[0])]).T
    beta_star = np.linalg.lstsq(X,Y)[0]
    classi = (lambda x : 1 if np.dot([x[0],x[1],1],beta_star)>=0.5 else 0)
    return classi, beta_star

# fonction d'affichage générique
def show_reg(tab, clas, name, dis):
    f, beta_star = clas(tab)
    show_bicolor(tab, name, False)
    b1, b2, b3 = beta_star[0],beta_star[1],beta_star[2]
    x=np.linspace(-4,4,100)
    if b2 != 0:
        pyplot.plot(x,1./b2 * (-b1*x-b3 + dis))
        pyplot.show()

# fonction d'affichage pour la régression linéaire
def show_reglin(tab, name):
    show_reg(tab, classifieur_reglin, name, 0.5)

show_reglin(trainA, 'Least square: A')    
show_reglin(trainB, 'Least square: B')
show_reglin(trainC, 'Least square: C')

def sigmoid(x):
    return 1. / (1 + math.exp(-x))

def l(x1,x2):
    return x2*math.log(1+math.exp(-x1)) + (1-x2)*math.log(1+math.exp(x1))

alpha = 0.1
eps = 0.05

def classifieur_reglog(tab):
    # Gradient descent instead of IRLS
    Y = tab.T[2]
    X = np.array([tab[:,0],tab[:,1], np.ones(tab[:,0].shape[0])]).T
    beta = np.zeros(3)
    n = len(Y)  
    def mu(beta):
        return np.array([sigmoid(np.dot(X[i],beta)) for i in range(n)])
    def f(beta):
        return (1./n)*np.sum([l(np.dot(X[i],beta),Y[i]) for i in range(n)])
    def gradf(beta):
        return (1./n)*np.dot(X.T,mu(beta)-Y)
    gradfb = gradf(beta)
    niter = 0
    # descente de gradient
    while np.linalg.norm(gradfb)>eps and niter < 666:
        beta = beta-alpha*gradfb
        gradfb = gradf(beta)
        niter += 1
    classi = (lambda x : 1 if np.dot([x[0],x[1],1],beta)>=0 else 0)
    return classi, beta

# fonction d'affichage pour la régression logistique
def show_reglog(tab, name):
    show_reg(tab, classifieur_reglog, name, 0)

show_reglog(trainA, 'Logistic: A')    
show_reglog(trainB, 'Logistic: B')
show_reglog(trainC, 'Logistic: C')


def classifieur_lda(tab):
    Y = tab.T[2]
    n = len(Y)
    X0 = np.array([[x[0],x[1]] for x in tab if x[2]==0])
    X1 = np.array([[x[0],x[1]] for x in tab if x[2]==1])
    X = np.array([tab[:,0],tab[:,1]]).T

    # calcul de mu_0
    mu_0 = [0,0]
    lx0 = len(X0)
    for i in range(lx0):
        mu_0 += X0[i]
    mu_0 = (1./lx0) * mu_0
    
    # calcul de mu_1
    mu_1 = [0,0]
    lx1 = len(X1)
    for i in range(lx1):
        mu_1 += X1[i]
    mu_1 = (1./lx1) * mu_1
            
    mu = [mu_0, mu_1]

    # calcul de sigma^-1
    in_sum = np.array([np.dot(np.array([(X[i]-mu[int(Y[i])])]).T,np.array([(X[i]-mu[int(Y[i])])])) 
                        for i in range(n)])
    sum_mat = [[0,0],[0,0]]
    for i in range(n):
        sum_mat += in_sum[i]
    sigma = (1./n) * sum_mat
    inv_sigma = np.linalg.inv(sigma)
    
    # constantes utilisées
    ppi = (1./ n) * np.sum(Y) # P(Y=1)
    cst = math.log((1 - ppi) / ppi)
    w = np.dot(inv_sigma, mu[1] - mu[0])
    b = (cst/2) * (np.dot(np.dot(mu[0].T,inv_sigma),mu[0])-np.dot(np.dot(mu[1].T,inv_sigma),mu[1])) - cst
    
    def classi(x):
        k = np.dot(w.T, x) + b
        return 1 if sigmoid(k) >= 0.5 else 0

    beta_star = np.hstack([w,b]) 
    return classi, beta_star

# fonction d'affichage pour Lda
def show_lda(tab, name):
    show_reg(tab, classifieur_lda, name, 0)

show_lda(trainA, 'LDA: A')    
show_lda(trainB, 'LDA: B')
show_lda(trainC, 'LDA: C')

def classifieur_qda(tab):
    Y = tab.T[2]
    n = len(Y)
    X0 = np.array([[x[0],x[1]] for x in tab if x[2]==0])
    X1 = np.array([[x[0],x[1]] for x in tab if x[2]==1])
    X = np.array([tab[:,0],tab[:,1]]).T

    # calcul de mu_0
    mu_0 = [0,0]
    lx0 = len(X0)
    for i in range(lx0):
        mu_0 += X0[i]
    mu_0 = (1./lx0) * mu_0
    
    # calcul de mu_1
    mu_1 = [0,0]
    lx1 = len(X1)
    for i in range(lx1):
        mu_1 += X1[i]
    mu_1 = (1./lx1) * mu_1
            
    mu = [mu_0, mu_1]
    
    # constante
    ppi = (1./ n) * lx1 # P(Y=1) = Π 

    # calcul de sigma_0
    sigma_0 = (1./lx0) * np.sum(np.array([np.dot(np.array([(X0[i]-mu[0])]).T,np.array([(X0[i]-mu[0])])) 
                                          for i in range(lx0)]), axis=0)
    inv_sigma_0 = np.linalg.inv(sigma_0)
    
    # calcul de sigma_1
    sigma_1 = (1./lx1) * np.sum(np.array([np.dot(np.array([(X1[i]-mu[1])]).T,np.array([(X1[i]-mu[1])])) 
                                   for i in range(lx1)]), axis=0)
    
    inv_sigma_1 = np.linalg.inv(sigma_1)
    
    # constantes utilisés
    musig_0 = np.dot(mu[0].T, inv_sigma_0)
    musig_1 = np.dot(mu[1].T, inv_sigma_1)
    factpi = math.log((1 - ppi) / ppi)
    factdet = math.log(math.sqrt(abs(np.linalg.det(sigma_1) / np.linalg.det(sigma_0))))
    
    def classi(x):        
        k = (-1./2)* (np.dot(np.dot(np.array(x), inv_sigma_0 - inv_sigma_1), np.array(x)) - 2*np.dot(musig_0,x) \
        + 2*np.dot(musig_1,x) + np.dot(musig_0,mu[0]) - np.dot(musig_1,mu[1])) + factpi + factdet
        return 1 if sigmoid(-k) >= 0.5 else 0
    
    # fonction utilisé pour l'affichage
    def classi_show(x):        
        k = (-1./2)* (np.dot(np.dot(np.array(x), inv_sigma_0 - inv_sigma_1), np.array(x)) - 2*np.dot(musig_0,x) \
        + 2*np.dot(musig_1,x) + np.dot(musig_0,mu[0]) - np.dot(musig_1,mu[1])) + factpi + factdet
        return sigmoid(-k)
    return classi, classi_show

# fonction d'affichage pour Qda
def show_qda(tab, name):
    _, f = classifieur_qda(tab)
    show_bicolor(tab, name, False)
    n = 400
    x = np.linspace(-6,7,n)
    y = np.linspace(-8,5,n)
    X,Y = np.meshgrid(x,y)
    def fs(x,y) : return f((x,y))
    Z = np.array([[ fs(X[i,j],Y[i,j]) for j in range(len(X[0]))] for i in range(len(X))])
    C = pyplot.contour(X, Y, Z, 1, colors='blue', linewidth=.5)
    #clabel(C, inline=1, fontsize=10)
    pyplot.show()

show_qda(trainA, 'QDA: A')    
show_qda(trainB, 'QDA: B')
show_qda(trainC, 'QDA: C')

def loss(x,y):
    return 0 if x==y else 1

# calcul du taux d'erreur
def taux_error(tabTrain,tabTest,cl):
    f,_ = cl(tabTrain)
    Y = tabTest.T[2]
    X = np.array([tabTest[:,0],tabTest[:,1]]).T
    n = len(Y)
    return (1./ n) * np.sum([loss(f(X[i]),Y[i]) for i in range(n)])
    
def show_error_class(cl,name, tr, te):
    print("  Entrainement sur train", name)
    print("    Test sur train",name, ":", float(int(10000*taux_error(tr,tr,cl)))/100, "%")
    print ("    Test sur test",name,":", float(int(10000*taux_error(tr,te,cl)))/100, "%")

EnsP = [("A", trainA, testA),("B", trainB, testB),("C", trainC, testC)]

def show_error(args):
    cl = args[0]
    clname = args[1]
    print("Taux erreur : Classifieur", clname)
    for f in EnsP:
        show_error_class(cl,f[0], f[1], f[2])
    print("")

EnsC = [(classifieur_reglin, "Least Square"),(classifieur_reglog, "Logistic"), \
        (classifieur_lda, "LDA"), (classifieur_qda, "QDA")]

list(map(show_error, EnsC))
