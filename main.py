import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
import matplotlib.ticker as ticker
import csv

class Physical_quantities:
    def __init__(self):
        self.q = 1.6e-19 # заряд в СИ
        self.k = 8.617e-5 # постоянная Больцмана в эВ
        self.h = 4.135e-15 
        self.m = 9.1e-31 # масса электрона в СИ
        self.k_si = 1.38e-23 # постоянная Больцмана в СИ

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
        return self.find_N_v(temperature) / 4 * np.exp(-1 * self.E_a / (self.k * temperature)) * (
            np.sqrt(1 + 8 * self.N_a / self.find_N_v(temperature) \
            * np.exp(self.E_a / self.k / temperature)) - 1)

    def n_2_3(self, temperature):
        return self.N_a / 2 * (1 + np.sqrt(1 + 4 * (self.find_ni(temperature) ** 2)/(self.N_a ** 2)))

    def find_temp_point(self):
        def func(temperature1_2):
            return self.n_1_2(temperature1_2) - self.n_2_3(temperature1_2)
        
        return fsolve(func, 400)[0]

    def find_fermi_i(self, temperature):
        return self.k * temperature * np.log((self.m_dp / self.m_dn) ** (3 / 4))

    def find_low_fermi(self, temperature):
        return -self.find_E_g(temperature) / 2 + self.k * temperature * \
            np.log(self.find_N_v(temperature) / ((self.find_N_v(temperature) * \
            np.exp(-self.E_a / (self.k * temperature)) / 2) * (-1 + np.sqrt(1 + \
            8 * (self.N_a / self.find_N_c(temperature)) * np.exp(self.E_a / (self.k * temperature))))))

    def find_high_fermi(self, temperature):
        return -self.find_E_g(temperature) / 2 + self.k * temperature *\
            np.log(self.find_N_v(temperature) / (self.N_a / 2 \
            * (1 + np.sqrt(1 + 4 * (self.find_ni(temperature) ** 2) / (self.N_a ** 2)))))

    def find_beta(self, temperature):
        return self.epsilond / 10 * temperature / 100 * (2.35e19 / self.N_a) ** (1 / 3)

    def find_mu_tepl(self, temperature):
        return self.mu_p * (temperature / 300) ** -1.5

    def find_mu_ion(self, temperature):
        return ((3.68e20 / self.N_a) * (self.epsilond / 16) ** 2) * ((temperature / 100) ** 1.5) * \
            ((self.m_dp ** 0.5) * np.log10(1 + self.find_beta(temperature) ** 2)) ** -1

    def find_mu(self, temperature):
        return (1 / self.find_mu_tepl(temperature) + 1 / self.find_mu_ion(temperature)) ** (-1)

    def find_n_critical(self, temperature):
        def find_V(temperature):
            return np.sqrt(3 * temperature * self.k_si / self.m_dp / self.m)

        def find_t(temperature):
            return self.find_mu(temperature) * 1e-4 * self.m_dp * self.m / self.q

        def find_l(temperature):
            return find_t(temperature) * find_V(temperature)

        return (np.pi * find_l(temperature) / 4) ** -3

    def find_Hall_ratio(self, temperature):
        if self.N_a * 1e6 > self.find_n_critical(temperature):
            return 1.93 / self.q / (self.N_a * 1e6)
        else:
            return 1.18 / self.q / (self.N_a * 1e6)

    def find_x(self, temperature):
        return self.xl_300 + 4 * self.k_si ** 2 * temperature / self.q * self.N_a * 1e6 \
            * self.find_mu(temperature) * 1e-4

    def find_termo_eds(self, temperature):
        return self.k_si / self.q * (5 / 2 + 3 / 2 + np.log(self.N_v_300 / self.N_a))

    def find_effect_ettingauzena(self, temperature):
        return 1 / 2 * self.k_si / self.q * temperature * self.q * self.N_a * 1e6 * \
            self.find_mu(300) * 1e-4 * self.find_termo_eds(temperature) / self.find_x(temperature)

class Input_param_Germanium(Сalculation_param_semicondactor, Physical_quantities):
    def __init__(self):
        Physical_quantities.__init__(self)
        self.m_dn = 0.56
        self.m_dp = 0.37
        self.g = 2
        self.mu_n = 3900
        self.mu_p = 1900
        self.epsilond = 16.3
        self.E_a = 0.0108
        self.N_a = 6e16
        self.N_v_300 = 6e18
        self.xl_300 = 60.2

class Draw_graph(Input_param_Germanium):
    def __init__(self):
        Input_param_Germanium.__init__(self)
        
    def draw_graph_n_t(self):
        temp_list = list()
        ln_n_list = list()

        for temperature in range (1, 1001):
            temp_list.append(1/temperature)
            if temperature <= self.find_temp_point():
                ln_n_list.append(np.log(self.n_1_2(temperature)) + 0.0191)
            else:
                ln_n_list.append(np.log(self.n_2_3(temperature)))

        fig, axes = plt.subplots()

        axes.plot(temp_list, ln_n_list, color='b', label='ln(p)', linewidth=1.5)
        axes.set(ylim=(37.5, 43))
        axes.set(xlim=(0, 0.024))

        plt.title('Температурная зависимость концентрации носителей заряда')
        plt.xlabel('1/T, 1/K')
        plt.ylabel('Ln(p)')
        plt.legend(loc=5)

        axes.xaxis.set_major_locator(ticker.MultipleLocator(0.004))
        axes.xaxis.set_minor_locator(ticker.MultipleLocator(0.001))
        axes.yaxis.set_major_locator(ticker.MultipleLocator(1))
        axes.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))

        axes.grid(which='major', color = '#666666')
        axes.minorticks_on()
        axes.grid(which='minor', color = 'gray', linestyle = ':')

        plt.show()

    def draw_graph_fermi(self):
        temp_list = list()
        E_c_list = list()
        E_v_list = list()
        E_a_list = list()
        F_list = list()
        Fi_list = list()

        for temperature in range(1, 1001):
            temp_list.append(temperature)
            E_c_list.append(self.find_E_g(temperature) / 2)
            E_v_list.append(-self.find_E_g(temperature) / 2)
            E_a_list.append(-self.find_E_g(temperature) / 2 + self.E_a)
            Fi_list.append(self.find_fermi_i(temperature))
            if temperature <= self.find_low_temperature():
                F_list.append(self.find_low_fermi(temperature))
            else:
                F_list.append(self.find_high_fermi(temperature))

        fig, axes = plt.subplots()

        axes.plot(temp_list, E_c_list, color='g', label='Ec(T)', linestyle='dotted', linewidth=2)
        axes.plot(temp_list, E_v_list, color='g', label='Ev(T)', linestyle='dashed', linewidth=1)
        axes.plot(temp_list, E_a_list, color='b', label='Ea(T)', linestyle='dashdot', linewidth=1)
        axes.plot(temp_list, Fi_list, color='y', label='Fi(T)', linestyle='dashed', linewidth=1)
        axes.plot(temp_list, F_list, color='r', label='F(T)', linestyle ='solid', linewidth=1)

        plt.title('Температурная зависимость положения уровня Ферми')
        axes.set(xlim=(1, 1000))
        plt.xlabel('T')
        fig.legend(loc='center left', borderaxespad=0.5)
        plt.subplots_adjust(left=0.2)        
    

        axes.xaxis.set_major_locator(ticker.MultipleLocator(200))
        axes.xaxis.set_minor_locator(ticker.MultipleLocator(20))
        axes.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        axes.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

        axes.grid(which='major', color = '#666666')
        axes.minorticks_on()
        axes.grid(which='minor', color = 'gray', linestyle = ':')

        plt.show()

    def draw_graph_mu(self):
        temp_list = list()
        mu_list = list()

        for temperature in range (1, 1001):
            temp_list.append(temperature)
            mu_list.append(self.find_mu(temperature))

        fig, axes = plt.subplots()
        axes.plot(temp_list, mu_list, color='g', linestyle='solid', label='µ(T)', linewidth=1.5)

        plt.title('Температурная зависимость подвижности носителей заряда')
        plt.xlabel('T, K')
        plt.ylabel('µ, См^2/(В*с)')
        axes.set(ylim=(0, 3600))
        axes.set(xlim=(0, 1000))
        plt.legend(loc=5)

        axes.xaxis.set_major_locator(ticker.MultipleLocator(150))
        axes.yaxis.set_major_locator(ticker.MultipleLocator(600))

        axes.grid(which='major', color = '#666666')
        axes.minorticks_on()
        axes.grid(which='minor', color = 'gray', linestyle = ':')

        plt.show()

    def draw_graph_conductivity(self):
        temp_list = list()
        conduct_list = list()

        for temperature in range(1, 1001):
            temp_list.append(1 / temperature)
            if temperature <= self.find_temp_point():
                conduct_list.append(np.log(self.n_1_2(temperature) * self.find_mu(temperature) * self.q))
            else:
                conduct_list.append(np.log(self.n_2_3(temperature) * self.find_mu(temperature) * self.q)\
                    - 0.0164)

        fig, axes = plt.subplots()

        axes.plot(temp_list, conduct_list, color='b', label='Ln(σ(T))', linewidth=1.5)
        axes.set(ylim=(1, 5.5))
        axes.set(xlim=(0, 0.02))

        plt.title('Температурная зависимость электропроводности')
        plt.xlabel('1/T, 1/K')
        plt.ylabel('Ln(σ)')
        plt.legend(loc=5)

        axes.xaxis.set_major_locator(ticker.MultipleLocator(0.003))
        axes.xaxis.set_minor_locator(ticker.MultipleLocator(0.0005))
        axes.yaxis.set_major_locator(ticker.MultipleLocator(1))
        axes.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))

        axes.grid(which='major', color = '#666666')
        axes.minorticks_on()
        axes.grid(which='minor', color = 'gray', linestyle = ':')

        plt.show()

class Create_table(Input_param_Germanium):
    def __init__(self):
        Input_param_Germanium.__init__(self)
        self.data_dict = {}
    
    def writer_table(self, data_dict):
        with open ('data_filik.csv', 'w', encoding='utf8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(('T', '1/T', 'Eg', 'ni', 'p', 'lnp', 'Ei+F(T)', 'Ei+Fi(T)',\
                'mu', 'sigma', 'ln(sigma)'))
            for i in range(len(self.data_dict['T'])):
                writer.writerow((self.data_dict['T'][i], self.data_dict['1/T'][i], \
                    self.data_dict['Eg'][i], self.data_dict['ni'][i],self.data_dict['p'][i], \
                    self.data_dict['lnp'][i], self.data_dict['Ei+F(T)'][i], \
                    self.data_dict['Ei+Fi(T)'][i], self.data_dict['mu'][i], \
                    self.data_dict['sigma'][i], self.data_dict['ln(sigma)'][i]))

    def table_for_lnp(self):
        table_n_list = list()
        table_temp_list = list()
        table_ln_n_list = list()
        table_rev_temp_list = list()

        for temperature in range (1, 1001):
            if temperature % 5 == 0:
                table_temp_list.append(temperature)
                table_rev_temp_list.append(1/temperature)
                if temperature <= self.find_temp_point():
                    table_n_list.append(self.n_1_2(temperature))
                    table_ln_n_list.append(np.log(self.n_1_2(temperature)))
                else:
                    table_n_list.append(self.n_2_3(temperature))
                    table_ln_n_list.append(np.log(self.n_2_3(temperature)))

        self.data_dict['T'] = table_temp_list
        self.data_dict['1/T'] = table_rev_temp_list
        self.data_dict['p'] = table_n_list
        self.data_dict['lnp'] = table_ln_n_list

    def table_for_fermi(self):
        table_F_list = list()
        table_Fi_list = list()

        for temperature in range(1, 1001):
            if temperature % 5 == 0:
                table_Fi_list.append(self.find_fermi_i(temperature))
                if temperature <= self.find_low_temperature():
                    table_F_list.append(self.find_low_fermi(temperature))
                else:
                    table_F_list.append(self.find_high_fermi(temperature)) 
        
        self.data_dict['Ei+F(T)'] = table_F_list
        self.data_dict['Ei+Fi(T)'] = table_Fi_list

    def table_for_mu(self):
        table_mu_list = list()

        for temperature in range (1, 1001):
            if temperature % 5 ==0:
                table_mu_list.append(self.find_mu(temperature))

        self.data_dict['mu'] = table_mu_list

    def table_for_conductivity(self):
        table_sigma_list = list()
        table_ln_sigma_list = list()

        for temperature in range(1, 1001):
            if temperature % 5 == 0:
                if temperature <= self.find_temp_point():
                    table_sigma_list.append(self.n_1_2(temperature) * \
                        self.find_mu(temperature) * self.q)
                    table_ln_sigma_list.append(np.log(self.n_1_2(temperature) * \
                        self.find_mu(temperature) * self.q))
                else:
                    table_sigma_list.append(self.n_2_3(temperature) * \
                        self.find_mu(temperature) * self.q)
                    table_ln_sigma_list.append(np.log(self.n_2_3(temperature) * \
                        self.find_mu(temperature) * self.q))

        self.data_dict['sigma'] = table_sigma_list
        self.data_dict['ln(sigma)'] = table_ln_sigma_list

    def table_for_ni_and_Eg(self):
        table_ni_list = list()
        table_Eg_list = list()

        for temperature in range (1, 1001):
            if temperature % 5 ==0:
                table_Eg_list.append(self.find_E_g(temperature))
                table_ni_list.append(self.find_ni(temperature))

        self.data_dict['Eg'] = table_Eg_list
        self.data_dict['ni'] = table_ni_list    

    def create_all_tables(self):
        self.table_for_lnp()
        self.table_for_fermi()
        self.table_for_mu()
        self.table_for_conductivity()
        self.table_for_ni_and_Eg()
        self.writer_table(self.data_dict)

def main():
    Ge = Draw_graph()
    Ge_table = Create_table()
    
    print('↓↓↓ Расчет параметров полупроводника ↓↓↓')
    print(f'1) Подвижность дырок при температуре 300К => {Ge.find_mu(300)} См^2/(B*c)')
    print(f'2) Удельная электропроводность при температуре 300К => {Ge.q * Ge.N_a * Ge.find_mu(300)} Ом * см')
    print(f'3) Коэффициент Холла при температуре 300К ==> {Ge.find_Hall_ratio(300)} м^3/Кл')
    print(f'4) Удельная теплопрводность при температуре 300К ==> {Ge.find_x(300)} Вт/(м*К)')
    print(f'5) Дифференциальная термо-эдс при температуре 300К ==> {Ge.find_termo_eds(300)} В/К')
    print(f'6) Эффект Эттинсгаузена при температуре 300К ==> {Ge.find_effect_ettingauzena(300)}')

    Ge.draw_graph_n_t()
    Ge.draw_graph_fermi()
    Ge.draw_graph_mu()
    Ge.draw_graph_conductivity()

    Ge_table.create_all_tables()

if __name__ == "__main__":
    main()