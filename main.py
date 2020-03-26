import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
import matplotlib.ticker as ticker

class Physical_quantities:
    def __init__(self):
        self.q = 1.6e-19 # заряд в СИ
        self.k = 8.617e-5 # постоянная Больцмана в эВ
        self.h = 4.135e-15

class Сalculation_param_semicondactor:
    def find_N_c(self, temperature):
        return 4.831e15 * (self.m_dn ** 1.5) * (temperature ** 1.5)
    
    def find_N_v(self, temperature):
        return 4.831e15 * (self.m_dp ** 1.5) * (temperature ** 1.5)

    def find_E_g(self, temperature):
        return 0.785 - 4.2e-4 * temperature

    def find_n(self, temperature):
        return ((self.find_N_c(temperature) * self.N_a / self.g) ** 0.5) * \
            np.exp(-self.E_a / 2 / self.k / temperature)

    def find_ni(self, temperature):
        return (self.find_N_c(temperature) * self.find_N_v(temperature)) ** 0.5 * \
            np.exp(-self.find_E_g(temperature) / 2 / self.k / temperature)
            
    def find_high_temperature(self):
        def high_func(high_temperature):
            return self.find_ni(high_temperature) - self.N_a * (2**0.5)
        high_temperature = fsolve(high_func, 800)
        return high_temperature[0]

    def find_low_temperature(self):
        def low_func(low_temperature):
            return self.N_a / self.find_N_c(low_temperature) * \
                np.exp(self.E_a / self.k / low_temperature) - 3
        low_temperature = fsolve(low_func, 50)
        return low_temperature[0]

    def n_1_2(self, temperature):
        #return 2 * self.N_a / (1 + (1 + ((4 * self.g * self.N_a / self.find_N_v(temperature)) * \
        #   np.exp((self.E_a / self.k / temperature))) ** 0.5))

        return self.find_N_v(temperature) / 4 * np.exp(-1 * self.E_a / (self.k * temperature)) * (
            np.sqrt(1 + 8 * self.N_a / self.find_N_v(temperature) \
            * np.exp(self.E_a / self.k / temperature)) - 1)

    def n_2_3(self, temperature):
        return self.N_a / 2 * (1 + np.sqrt(1 + 4 * (self.find_ni(temperature) ** 2)/(self.N_a ** 2)))

    def find_temp_point(self):
        def func(temperature1_2):
            return self.n_1_2(temperature1_2) - self.n_2_3(temperature1_2)
        
        return fsolve(func, 400)[0]

class Input_param_Germanium(Сalculation_param_semicondactor, Physical_quantities):
    def __init__(self):
        Physical_quantities.__init__(self)
        self.m_dn = 0.56
        self.m_dp = 0.37
        self.g = 2
        self.E_g_300 = 0.67
        self.mu_n = 3900
        self.mu_p = 1900
        self.n_i_300 = 2.4e13
        self.N_c_300 = 1.04e19
        self.N_v_300 = 6e18
        self.epsilond = 16.3
        self.E_a = 0.0108
        self.N_a = 6e16

class Graph_draw(Input_param_Germanium):
    def __init__(self):
        Input_param_Germanium.__init__(self)

    def draw_graph_n_t(self):
        temp_list = list()
        ln_n_list = list()
        for temperature in range (2, 1002):
            temp_list.append(1/temperature)
            if temperature <= 400:
                ln_n_list.append(np.log(self.n_1_2(temperature)) + 0.0191)
            else:
                ln_n_list.append(np.log(self.n_2_3(temperature)))

        fig, axes = plt.subplots()

        axes.plot(temp_list, ln_n_list, color='b', label='ln(p)', linewidth=1.5)
        axes.set(ylim=(37.5, 43))
        axes.set(xlim=(0, 0.025))

        plt.title('Температурная зависимость концентрации носителей заряда')
        plt.xlabel('1/T, 1/K')
        plt.ylabel('Ln(p)')
        plt.legend(loc=5)

        axes.xaxis.set_major_locator(ticker.MultipleLocator(0.005))
        axes.xaxis.set_minor_locator(ticker.MultipleLocator(0.001))
        axes.yaxis.set_major_locator(ticker.MultipleLocator(1))
        axes.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))

        axes.grid(which='major', color = '#666666')
        axes.minorticks_on()
        axes.grid(which='minor', color = 'gray', linestyle = ':')

        plt.show()

    def draw_graph_fermi(self):
        pass

def main():
    Ge = Graph_draw()
    Ge.draw_graph_n_t()

if __name__ == "__main__":
    main()