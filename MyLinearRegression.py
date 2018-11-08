import matplotlib.pyplot as plt
import pandas

def find_gradient(x,y,m,b):
    diffm=[]
    diffb=[]
    for i in range(len(x)):
        diffb.append(y[i]-(m*x[i]+b))
        diffm.append(x[i]*(y[i]-m*x[i]+b))
    gradientm=sum(diffm)/len(x)
    gradientb=sum(diffb)/len(x)
    return gradientm,gradientb
def gradient_descent(x,y,m,b,learning_rate):
    change_m,change_b=find_gradient(x,y,m,b)
    m=m+learning_rate*change_m
    b=b+learning_rate*change_b
    return m,b
def main():
    df=pandas.read_csv('train.csv').dropna()
    x=df['x'].tolist()
    y=df['y'].tolist()
    num_iterations=1000
    learning_rate=0.00001
    m=0
    b=0
    for i in range(num_iterations):
        m,b=gradient_descent(x,y,m,b,learning_rate)
    plt.plot(x,y,"o")

    y_predicted=[]
    for i in range(len(x)):
        y_predicted.append(m*x[i]+b)
    plt.plot(x,y_predicted)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression')
    plt.show()
main()

