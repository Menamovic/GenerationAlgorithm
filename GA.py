import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sympy import *

DNA_SIZE = 20
POP_SIZE = 200
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.005
N_GENERATIONS = 500
P_BOUND = [0, 100]
N3 = 1e10
N4 = 1e10

#def F(x, y):
#    return 3*(1-x)**2*np.exp(-(x**2)-(y+1)**2)- 10*(x/5 - x**3 - y**5)*np.exp(-x**2-y**2)- 1/3**np.exp(-(x+1)**2 - y**2)

def F(p) :
    t = symbols('t')
    _p = []
    for i in range(len(p)) :
        _p.append(37.86 * integrate(N3 * math.e**(-(0.3 + 0.42 * p[i]) * t) * 0.42 * p[i], (t, 0, 243)) \
            + 50.99 * integrate(N4 * p[i] * math.e**(-(0.3 + p[i]) * t), (t, 0, 243)))
    _p = np.array(_p)
    _p.astype('float64')
    return _p
'''
def plot_3d(ax):
    X = np.linspace(*X_BOUND, 100)
    Y = np.linspace(*Y_BOUND, 100)
    X,Y = np.meshgrid(X, Y)
    Z = F(X, Y)
    ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap=cm.coolwarm)
    ax.set_zlim(-10,10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.pause(3)
    plt.show()
'''

def plot_2d() :
    P = np.linspace(*P_BOUND, 100)
    Z = F(P)
    plt.grid(True)
    plt.plot(P,Z,label='Z-P')
    plt.plot([2*min(P),2*max(P)],[0,0],label='p-axis')
    plt.plot([0,0],[2*min(Z),3*max(Z)],label='z-axis')
'''
def get_fitness(pop):
    x,y = translateDNA(pop)
    pred = F(x, y)
    return (pred - np.min(pred)) #减去最小的适应度是为了防止适应度出现负数，通过这一步fitness的范围为[0, np.max(pred)-np.min(pred)]
'''

def get_fitness(pop) :
    p = translateDNA(pop)
    pred = F(p)
    pred = pred.astype('float64')
    return pred

'''
def translateDNA(pop): #pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
    x_pop = pop[:,1::2]#奇数列表示X,从索引列1开始，加入了步长2
    y_pop = pop[:,::2] #偶数列表示y,从索引列1开始，加入了步长2

    #pop:(POP_SIZE,DNA_SIZE)*(DNA_SIZE,1) --> (POP_SIZE,1)
    x = x_pop.dot(2**np.arange(DNA_SIZE)[::-1])/float(2**DNA_SIZE-1)*(X_BOUND[1]-X_BOUND[0])+X_BOUND[0]
    y = y_pop.dot(2**np.arange(DNA_SIZE)[::-1])/float(2**DNA_SIZE-1)*(Y_BOUND[1]-Y_BOUND[0])+Y_BOUND[0]
    return x,y
'''

def translateDNA(pop) :
    p_pop = pop.dot(2**np.arange(DNA_SIZE)[::-1])   #   二进制转十进制
    p_pop = p_pop / float(2**DNA_SIZE - 1)  # 将实数压缩到[0,1]
    p = p_pop * (P_BOUND[1] - P_BOUND[0]) + P_BOUND[0]  #映射为p范围内的数
    return p

def crossover_and_mutation(pop, CROSSOVER_RATE = 0.8):
    new_pop = []
    for father in pop:		#遍历种群中的每一个个体，将该个体作为父亲
        child = father		#孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
        if np.random.rand() < CROSSOVER_RATE:			#产生子代时不是必然发生交叉，而是以一定的概率发生交叉
            mother = pop[np.random.randint(POP_SIZE)]	#再种群中选择另一个个体，并将该个体作为母亲
            cross_points = np.random.randint(low=0, high=DNA_SIZE)	#随机产生交叉的点
            child[cross_points:] = mother[cross_points:]		#孩子得到位于交叉点后的母亲的基因
        mutation(child)	#每个后代有一定的机率发生变异
        new_pop.append(child)

    return new_pop

def mutation(child, MUTATION_RATE=0.003):
    if np.random.rand() < MUTATION_RATE: 				#以MUTATION_RATE的概率进行变异
        mutate_point = np.random.randint(0, DNA_SIZE)	#随机产生一个实数，代表要变异基因的位置
        child[mutate_point] = child[mutate_point]^1 	#将变异点的二进制为反转

def select(pop, fitness):    #  返回被选中的个体(轮盘赌法)
    #   np.random.choice : POP_SIZE个个体中，根据每个个体被选择的概率，返回被选中的个体
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=(fitness)/(fitness.sum()) )
    return pop[idx]

def print_info(pop):
    fitness = get_fitness(pop)
    max_fitness_index = np.argmax(fitness)
    print("max_fitness:", fitness[max_fitness_index])
    p = translateDNA(pop)
    print("最优的基因型：", pop[max_fitness_index])
    print("p:", p[max_fitness_index])

if __name__ == "__main__":
    fig = plt.figure()
    #ax = Axes3D(fig)
    plt.ion()#将画图模式改为交互模式，程序遇到plt.show不会暂停，而是继续执行
    plot_2d()

    #   初始种群
    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE)) #matrix (POP_SIZE, DNA_SIZE)
    #   迭代N代
    for _ in range(N_GENERATIONS):
        print("Generation", _, ":")
        #   解码
        p = translateDNA(pop)
        #   画图
        if 'sca' in locals():
            sca.remove()
        sca = plt.scatter(p, F(p), c='black', marker='o')
        plt.show()
        plt.pause(0.05)
        #   输出当前代的信息
        print_info(pop)
        #   交叉和变异
        pop = np.array(crossover_and_mutation(pop, CROSSOVER_RATE))
        #   适应度
        fitness = get_fitness(pop)
        #   选择生成新的种群
        pop = select(pop, fitness)
    #   输出最终代的信息
    print_info(pop)
    plt.ioff()
    #plot_3d(ax)
    plot_2d()
