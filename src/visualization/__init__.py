
def make_line_plot(ax, df, x_col, y_col, 
        title='', 
        ylabel='',
        xlabel='',
        ):
    for col in y_col:
        ax.plot(df[col], label=col)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if len(y_col) > 1:
        ax.legend(bbox_to_anchor=(1.25, 1.05))

# events = { label: array_of_indices }
def mark_event(ax, events):

    color_sets = ['black', 'magenta', 'yellow', 'green']
    color_index = 0

    for event_name, indices in events.iteritems():
        curr_color = color_sets[color_index]
        for i in indices:
            ax.axvline(x=i, linewidth=1, color=curr_color, linestyle='dashed', label=event_name)
        color_index += 1

    # stop matplotlib from repeating labels in legend for axvline
    handles, labels = ax.get_legend_handles_labels()
    handle_list, label_list = [], []
    for handle, label in zip(handles, labels):
        if label not in label_list:
            handle_list.append(handle)
            label_list.append(label)
    
    ax.legend(handle_list, label_list, fontsize=8, loc='upper right', bbox_to_anchor=(1.28, 1.03))
