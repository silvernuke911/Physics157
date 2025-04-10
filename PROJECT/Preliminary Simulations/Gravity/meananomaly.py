import matplotlib.pyplot as plt
import math as mt

def eccentric_anomaly(true_anomaly):
    if deg==False:
        n=eccentricity+mt.cos(true_anomaly)
        d=1+eccentricity*mt.cos(true_anomaly)
        if true_anomaly>=mt.pi:
            return 2*mt.pi-mt.acos(n/d)
        return mt.acos(n/d)
    if deg==True:
        true_anomaly=deg_rad_convertar(true_anomaly,'degtorad')
        n=eccentricity+mt.cos(true_anomaly)
        d=1+eccentricity*mt.cos(true_anomaly)
        if true_anomaly>=mt.pi:
            return deg_rad_convertar(2*mt.pi-mt.acos(n/d),"radtodeg")
        return deg_rad_convertar(mt.acos(n/d),"radtodeg")
def mean_anomaly(true_anomaly):
    if deg==False:
        return eccentric_anomaly(true_anomaly)-eccentricity*mt.sin(eccentric_anomaly(true_anomaly))
    if deg==True:
        mean_a=deg_rad_convertar(eccentric_anomaly(true_anomaly),"degtorad")-eccentricity*mt.sin(deg_rad_convertar(eccentric_anomaly(true_anomaly),"degtorad"))
        return deg_rad_convertar(mean_a,"radtodeg")
def true_anomaly_list(division):
    anomalylist=[]
    if deg==False:
        n=2*mt.pi/division
    if deg==True:
        n=360/division
    m=-n
    for i in range(division+1):
        m+=n
        anomalylist.append(m)
    return anomalylist
def model_anomaly(mean_anomaly):
    if deg==False:
        numer=mt.cos(mean_anomaly)+(eccentricity**2)*mt.cos(mean_anomaly)-2*eccentricity
        denom=1+eccentricity**2-2*eccentricity*mt.cos(mean_anomaly)
        if mean_anomaly>mt.pi:
            return 2*mt.pi-mt.acos(numer/denom)
        return mt.acos(numer/denom)
    if deg==True:
        mean_anomaly=deg_rad_convertar(mean_anomaly,"degtorad")
        numer=mt.cos(mean_anomaly)+(eccentricity**2)*mt.cos(mean_anomaly)-2*eccentricity
        denom=1+eccentricity**2-2*eccentricity*mt.cos(mean_anomaly)
        if mean_anomaly>mt.pi:
            return 360-deg_rad_convertar(mt.acos(numer/denom),"radtodeg")
        return deg_rad_convertar(mt.acos(numer/denom),"radtodeg")
def function_list_maker(x_list,f_x):
    output=[]
    for i in range(len(x_list)):
        output.append(f_x(x_list[i]))
    return output
def difference_of_list(list1,list2):
    n=len(list1)
    output=[]
    if len(list2)!=n:
        return "value not equal"
    for i in range(n):
        m=list1[i]-list2[i]
        output.append(m)
    return(output)
def deg_rad_convertar(angle,mode):
    if mode=="radtodeg":
        return angle*180/mt.pi
    if mode=="degtorad":
        return angle*mt.pi/180
def plot_anomaly():
    if deg==False:
        plt.axis([-0,2*mt.pi,-0.5,2*mt.pi])
    if deg==True:
        plt.axis([0,360,0,360])
    plt.grid()
    #plt.scatter(init_angle_list,eccentric_anomaly_list,marker=".",color='red')
    plt.plot(mean_anomaly_list,init_angle_list,marker="",linestyle="-",color='blue')
    plt.plot(mean_anomaly_list,model_anomaly_list,marker="",linestyle="-",color='red')
    plt.plot(mean_anomaly_list,difference_list,marker="",linestyle="-",color='green')
    plt.plot(mean_anomaly_list,newton_iteration_list,marker="",linestyle="-",color='cyan')
    plt.show()
def Newton_iteration(mean_anomaly):
    def eccentric_from_mean(mean_anomaly):
        max_iterations=100
        tolerance=10e-9
        if deg==False:
            E=mean_anomaly
        if deg==True:
            E=deg_rad_convertar(mean_anomaly,'degtorad')
            mean_anomaly=deg_rad_convertar(mean_anomaly,'degtorad')
        for _ in range(max_iterations):
            E_new = E - (E - eccentricity * mt.sin(E) - mean_anomaly) / (1 - eccentricity * mt.cos(E))
            if abs(E_new - E) < tolerance:
                return E_new
            E = E_new
        raise ValueError("Eccentric anomaly calculation did not converge")
    def true_from_eccentric(E):
        return 2 * mt.atan2(mt.sqrt(1 + eccentricity) * mt.sin(E/2),
                           mt.sqrt(1 - eccentricity) * mt.cos(E/2))
    E=eccentric_from_mean(mean_anomaly)
    m=true_from_eccentric(E)
    if deg==True:
        return deg_rad_convertar(m,"radtodeg")
    else:
        return m

eccentricity=0.77
deg=False

init_angle_list=true_anomaly_list(3600)
eccentric_anomaly_list=function_list_maker(init_angle_list,eccentric_anomaly)
mean_anomaly_list=function_list_maker(init_angle_list,mean_anomaly)
model_anomaly_list=function_list_maker(mean_anomaly_list,model_anomaly)
difference_list=difference_of_list(init_angle_list,model_anomaly_list)
newton_iteration_list=function_list_maker(mean_anomaly_list,Newton_iteration)
difference_list2=difference_of_list(init_angle_list,newton_iteration_list)
for _ in range(len(init_angle_list)):
    print(mean_anomaly_list[_],init_angle_list[_],newton_iteration_list[_],model_anomaly_list[_],difference_list2[_],difference_list[_])
plot_anomaly()


# for i in range(10):   newton_iteration_list[_],difference_list2[_],difference_list[_]
#     eccentricity+=0.1
#     init_angle_list=true_anomaly_list(360)
#     eccentric_anomaly_list=function_list_maker(init_angle_list,eccentric_anomaly)
#     mean_anomaly_list=function_list_maker(init_angle_list,mean_anomaly)
#     model_anomaly_list=function_list_maker(mean_anomaly_list,model_anomaly)
#     difference_list=difference_of_list(init_angle_list,model_anomaly_list)
#     plot_anomaly()
 
#honestly, I dont know why I'm doing this, the difference from newton iteration is almost zero, its crazy.
#computers are fast, jesus
#perhaps a new numerical method, idk
#for loops are goated