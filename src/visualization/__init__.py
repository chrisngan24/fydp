import matplotlib.pyplot as plt

def make_line_plot(df, x_col, y_col, 
        title='', 
        file_dir= '',
        ylabel='',
        xlabel='',
        ):
    plt.style.use('ggplot')
    plt.figure()
    for col in y_col:
        plt.plot(df[x_col], df[col], label=col)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if len(y_col) > 1:
        plt.legend()
        #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    plt.savefig('%s/%s-%s.png' % (file_dir, x_col, y_col))
