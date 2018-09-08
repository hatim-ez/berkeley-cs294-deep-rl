import matplotlib.pyplot as plt

def main():
    # The iteration number 0 corresponds to 0 DAger's iteration, which corresponds to the simple behavioral cloning model
    iterations = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    dagger_mean_return = [499, 491.00752037626864, 280.35493455713464, 313.0841462553591, 281.2280666853588, 302.32458150840415, 375.3346917930867,
                                526.2328895242667, 722.7978505443548, 623.9050590062458, 1398.44995600636, 3076.955646121361, 8253.923134108587,
                                7629.325002257281, 7246.285399942843, 8087.738743306754, 3893.1664238204467, 9132.88474770422, 9460.28598069521,
                                9726.914701058917]
    dagger_stds = [499, 107.8460299321844, 13.621758933207397, 13.659271852314815, 18.423331558439394, 9.814398412282554, 184.14841961543374,
                    207.72978567273103, 508.0083698513861, 695.3579276517936, 904.658395758084, 2612.8369380474232, 3498.691955383935,
                    3287.0477377140114, 4022.94659430406, 2843.4029358558473, 1324.384546716151, 2446.9232315473655, 2682.070975182269,
                    2073.0612932329764]

    plt.figure()

    plt.errorbar(iterations, dagger_mean_return, color='r', yerr=dagger_stds, label='DAgger')
    plt.suptitle('DAgger - Mean reward function of number of iterations')
    plt.xlabel('DAgger iteration')
    plt.ylabel('Mean reward')
    plt.xlim([-0.5, 20])
    plt.ylim([450, 10800])
    expert = plt.axhline(y=10477, color='g', label='Expert Policy')
    bc = plt.axhline(y=499, color='b', label='Behavioral Cloning')
    plt.legend(loc=4)
    plt.savefig('figures/Q3.2.png')
    plt.show()


if __name__ == '__main__':
    main()