from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from plotly import tools


def plot_mlp_supervised_bar(mlp_mean_W_5_metadata, mlp_frac_decay_W_5_metadata, mlp_exp_decay_W_5_metadata):
    trace1 = go.Bar(
        x=['MLP.1 - promedio', 'MLP.2 - decaimiento fraccional', 'MLP.3 - decaimiento exponencial'],
        y=[mlp_mean_W_5_metadata['acc'][99],
           mlp_frac_decay_W_5_metadata['acc'][99],
           mlp_exp_decay_W_5_metadata['acc'][99]],
        name='Entrenamiento'
    )
    trace2 = go.Bar(
        x=['MLP.1 - promedio', 'MLP.2 - decaimiento fraccional', 'MLP.3 - decaimiento exponencial'],
        y=[mlp_mean_W_5_metadata['val_acc'][99],
           mlp_frac_decay_W_5_metadata['val_acc'][99],
           mlp_exp_decay_W_5_metadata['val_acc'][99]],
        name='Validación'
    )
    trace3 = go.Bar(
        x=['MLP.1 - promedio', 'MLP.2 - decaimiento fraccional', 'MLP.3 - decaimiento exponencial'],
        y=[mlp_mean_W_5_metadata['test_acc'],
           mlp_frac_decay_W_5_metadata['test_acc'],
           mlp_exp_decay_W_5_metadata['test_acc']],
        name='Test'
    )
    data = [trace1, trace2, trace3]
    layout = go.Layout(
        barmode='group',
        title='Baselines Perceptrón Multicapa',
        xaxis=dict(
            title='estrategia'
        ),
        yaxis=dict(
            title='exactitud (accuracy)'
        )
    )
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)


def plot_cnn_wide_supervised(df_list, title, yrange, metric, rows=2, cols=2):
    all_traces = []
    for idx, df in enumerate(df_list):
        epochs = df['epoch'].values
        if metric == 'acc':
            train_metric = df['acc'].values
            val_metric = df['val_acc'].values
        elif metric == 'loss':
            train_metric = df['loss'].values
            val_metric = df['val_loss'].values
        idx = str(idx + 1)
        trace0 = go.Scatter(
            x=epochs,
            y=train_metric,
            marker=dict(color='#0099ff'),
            xaxis='x' + idx,
            yaxis='y' + idx,
            name='entrenamiento'
        )
        trace1 = go.Scatter(
            x=epochs,
            y=val_metric,
            marker=dict(color='#fc8a00'),
            xaxis='x' + idx,
            yaxis='y' + idx,
            name='validación'
        )
        all_traces += [trace0, trace1]
    figure = tools.make_subplots(rows=rows, cols=cols,
                                 subplot_titles=('25% instancias entrenamiento',
                                                 '50% instancias entrenamiento',
                                                 '75% instancias entrenamiento',
                                                 '100% instancias entrenamiento'))
    figure.layout.showlegend = False
    figure.layout.yaxis.range = yrange
    figure.layout.yaxis2.range = yrange
    figure.layout.yaxis3.range = yrange
    figure.layout.yaxis4.range = yrange
    figure.layout.title = title
    _ = figure.add_traces(all_traces)
    if metric == 'acc':
        figure['layout']['xaxis1'].update(title='épocas')
        figure['layout']['xaxis2'].update(title='épocas')
        figure['layout']['xaxis3'].update(title='épocas')
        figure['layout']['xaxis4'].update(title='épocas')
        figure['layout']['yaxis1'].update(title='exactitud (accuracy)', dtick=0.05)
        figure['layout']['yaxis2'].update(title='exactitud (accuracy)', dtick=0.05)
        figure['layout']['yaxis3'].update(title='exactitud (accuracy)', dtick=0.05)
        figure['layout']['yaxis4'].update(title='exactitud (accuracy)', dtick=0.05)
    elif metric == 'loss':
        figure['layout']['xaxis1'].update(title='épocas')
        figure['layout']['xaxis2'].update(title='épocas')
        figure['layout']['xaxis3'].update(title='épocas')
        figure['layout']['xaxis4'].update(title='épocas')  # , type='log')
        figure['layout']['yaxis1'].update(title='pérdida (loss)', dtick=0.5)
        figure['layout']['yaxis2'].update(title='pérdida (loss)', dtick=0.5)
        figure['layout']['yaxis3'].update(title='pérdida (loss)', dtick=0.5)
        figure['layout']['yaxis4'].update(title='pérdida (loss)', dtick=0.5)
    iplot(figure)


def plot_cnn_depth_supervised(df_list, title, yrange, metric, rows=3, cols=3):
    all_traces = []
    for idx, df in enumerate(df_list):
        epochs = df['epoch'].values
        if metric == 'acc':
            train_metric = df['acc'].values
            val_metric = df['val_acc'].values
        elif metric == 'loss':
            train_metric = df['loss'].values
            val_metric = df['val_loss'].values
        idx = str(idx + 1)
        trace0 = go.Scatter(
            x=epochs,
            y=train_metric,
            marker=dict(color='#0099ff'),
            xaxis='x' + idx,
            yaxis='y' + idx,
            name='entrenamiento'
        )
        trace1 = go.Scatter(
            x=epochs,
            y=val_metric,
            marker=dict(color='#fc8a00'),
            xaxis='x' + idx,
            yaxis='y' + idx,
            name='validación'
        )
        all_traces += [trace0, trace1]
    figure = tools.make_subplots(rows=rows, cols=cols,
                                 subplot_titles=('3 capas conv | 50 filtros',
                                                 '10 capas conv | 50 filtros',
                                                 '3 capas conv | 100 filtros',
                                                 '5 capas conv | 100 filtros'))
    figure.layout.showlegend = False
    figure.layout.yaxis.range = yrange
    figure.layout.yaxis2.range = yrange
    figure.layout.yaxis3.range = yrange
    figure.layout.yaxis4.range = yrange
    # figure.layout.yaxis5.range = yrange
    # figure.layout.yaxis6.range = yrange
    figure.layout.title = title
    _ = figure.add_traces(all_traces)
    if metric == 'acc':
        figure['layout']['xaxis1'].update(title='épocas')
        figure['layout']['xaxis2'].update(title='épocas')
        figure['layout']['xaxis3'].update(title='épocas')
        figure['layout']['xaxis4'].update(title='épocas')
        # figure['layout']['xaxis5'].update(title='épocas')
        # figure['layout']['xaxis6'].update(title='épocas')
        figure['layout']['yaxis1'].update(title='exactitud (accuracy)', dtick=0.05)
        figure['layout']['yaxis2'].update(title='exactitud (accuracy)', dtick=0.05)
        figure['layout']['yaxis3'].update(title='exactitud (accuracy)', dtick=0.05)
        figure['layout']['yaxis4'].update(title='exactitud (accuracy)', dtick=0.05)
        # figure['layout']['yaxis5'].update(title='exactitud')
        # figure['layout']['yaxis6'].update(title='exactitud')
    elif metric == 'loss':
        figure['layout']['xaxis1'].update(title='épocas')
        figure['layout']['xaxis2'].update(title='épocas')
        figure['layout']['xaxis3'].update(title='épocas')
        figure['layout']['xaxis4'].update(title='épocas')
        # figure['layout']['xaxis5'].update(title='épocas')
        # figure['layout']['xaxis6'].update(title='épocas')
        figure['layout']['yaxis1'].update(title='pérdida (loss)')
        figure['layout']['yaxis2'].update(title='pérdida (loss)')
        figure['layout']['yaxis3'].update(title='pérdida (loss)')
        figure['layout']['yaxis4'].update(title='pérdida (loss)')
        # figure['layout']['yaxis5'].update(title='pérdida (loss)')
        # figure['layout']['yaxis6'].update(title='pérdida (loss)')
    iplot(figure)



def plot_best_supervised_bar(mlp_dict, cnn_wide_dict, cnn_depth_dict):

    trace1 = go.Bar(
        x=['MLP.3 - decaimiento exponencial', 'cnn wide', 'cnn depth'],
        y=[mlp_dict['train_acc'],
           cnn_wide_dict['train_acc'],
           cnn_depth_dict['train_acc']],
        name='Entrenamiento'
    )
    trace2 = go.Bar(
        x=['MLP.3 - decaimiento exponencial', 'cnn wide', 'cnn depth'],
        y=[mlp_dict['val_acc'],
           cnn_wide_dict['val_acc'],
           cnn_depth_dict['val_acc']],
        name='Validación'
    )
    trace3 = go.Bar(
        x=['MLP.3 - decaimiento exponencial', 'cnn wide', 'cnn depth'],
        y=[mlp_dict['test_acc'],
           cnn_wide_dict['test_acc'],
           cnn_depth_dict['test_acc']],
        name='Test'
    )
    data = [trace1, trace2, trace3]
    layout = go.Layout(
        barmode='group',
        title='Mejores modelos baseline supervisado',
        xaxis=dict(
            title='modelo'
        ),
        yaxis=dict(
            title='exactitud (accuracy)'
        )
    )
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)  
