import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
import dash
from dash import dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Data 
df = pd.read_csv('data/processed/cookie_cats_cleaned.csv')

COLOR_MAP     = {'gate_30': '#7C3AED', 'gate_40': '#F59E0B'}
SEGMENT_COLOR = {'Casual': '#7C3AED', 'Regular': '#F59E0B', 'Hardcore': '#10B981'}
SEGMENT_ORDER = ['Casual', 'Regular', 'Hardcore']
SEED = 42
np.random.seed(SEED)

CARD_BG      = 'rgba(255,255,255,0.03)'
CARD_BORDER  = 'rgba(124,58,237,0.18)'
GRID_COLOR   = 'rgba(255,255,255,0.06)'
FONT         = 'DM Sans, sans-serif'
MONO         = 'DM Mono, monospace'
TEXT_PRIMARY = 'rgba(232,230,240,0.85)'
TEXT_DIM     = 'rgba(232,230,240,0.45)'

PLOT_BASE = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color=TEXT_PRIMARY, family=FONT),
    title_font=dict(color='rgba(232,230,240,0.9)', size=14),
    legend=dict(font=dict(color=TEXT_DIM)),
    margin=dict(t=48, b=36, l=44, r=20),
    xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor='rgba(255,255,255,0.08)',
               linecolor='rgba(255,255,255,0.08)'),
    yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor='rgba(255,255,255,0.08)',
               linecolor='rgba(255,255,255,0.08)'),
)

# Pre-compute all data

# Group balance
group_counts = df['version'].value_counts().reset_index()
group_counts.columns = ['version', 'count']

# Engagement
df_hist = df[df['sum_gamerounds'] <= 100]

# Retention
retention = (
    df.groupby('version')[['retention_1', 'retention_7']]
    .mean().mul(100).round(2).reset_index()
)

# Segment
seg_dist = df['segment'].value_counts().reset_index()
seg_dist.columns = ['segment', 'count']
seg_dist['share'] = (seg_dist['count'] / len(df) * 100).round(1)
seg_sorted = seg_dist.set_index('segment').reindex(['Hardcore', 'Regular', 'Casual']).reset_index()

seg_ret = (
    df.groupby(['segment', 'version'])[['retention_1', 'retention_7']]
    .mean().mul(100).round(2).reset_index()
)

# Bootstrap
N_ITER = 10_000
n      = len(df)
ret1   = df['retention_1'].values
ret7   = df['retention_7'].values
vers   = df['version'].values
is30   = vers == 'gate_30'

b1 = {'gate_30': [], 'gate_40': []}
b7 = {'gate_30': [], 'gate_40': []}
for _ in range(N_ITER):
    idx = np.random.randint(0, n, size=n)
    m30 = is30[idx]
    b1['gate_30'].append(ret1[idx][m30].mean())
    b1['gate_40'].append(ret1[idx][~m30].mean())
    b7['gate_30'].append(ret7[idx][m30].mean())
    b7['gate_40'].append(ret7[idx][~m30].mean())

boot1 = pd.DataFrame(b1)
boot7 = pd.DataFrame(b7)
boot1['diff'] = (boot1['gate_30'] - boot1['gate_40']) / boot1['gate_40'] * 100
boot7['diff'] = (boot7['gate_30'] - boot7['gate_40']) / boot7['gate_40'] * 100

prob_1d = (boot1['diff'] > 0).mean()
prob_7d = (boot7['diff'] > 0).mean()

# Z-test
def run_ztest(metric):
    succ = df.groupby('version')[metric].sum()
    nobs = df.groupby('version')[metric].count()
    counts = [succ['gate_30'], succ['gate_40']]
    ns     = [nobs['gate_30'], nobs['gate_40']]
    z, p   = proportions_ztest(count=counts, nobs=ns, alternative='larger')
    ci30   = proportion_confint(counts[0], ns[0], alpha=0.05, method='wilson')
    ci40   = proportion_confint(counts[1], ns[1], alpha=0.05, method='wilson')
    return {
        'rate_30': counts[0]/ns[0]*100, 'ci_30': (ci30[0]*100, ci30[1]*100),
        'rate_40': counts[1]/ns[1]*100, 'ci_40': (ci40[0]*100, ci40[1]*100),
        'z': round(z,4), 'p': round(p,6), 'sig': p < 0.05
    }

r1 = run_ztest('retention_1')
r7 = run_ztest('retention_7')

def cohens_h(p1, p2):
    return abs(2*np.arcsin(np.sqrt(p1)) - 2*np.arcsin(np.sqrt(p2)))

h1 = cohens_h(r1['rate_30']/100, r1['rate_40']/100)
h7 = cohens_h(r7['rate_30']/100, r7['rate_40']/100)

# Build figures

# 1. Group balance
fig_balance = go.Figure()
for _, row in group_counts.iterrows():
    fig_balance.add_trace(go.Bar(
        x=[row['version']], y=[row['count']],
        name=row['version'], marker_color=COLOR_MAP[row['version']],
        text=[f"{row['count']:,}"], textposition='outside'
    ))
fig_balance.update_layout(
    **PLOT_BASE, title='Player Count per A/B Group', showlegend=False,
    yaxis_range=[0, group_counts['count'].max() * 1.15],
    xaxis_title='A/B Group', yaxis_title='Players'
)

# 2. Engagement histogram
counts_hist, bins = np.histogram(df_hist['sum_gamerounds'], bins=100)
bin_centers = (bins[:-1] + bins[1:]) / 2

fig_hist = go.Figure()
fig_hist.add_trace(go.Bar(
    x=bin_centers, y=counts_hist,
    marker_color='#7C3AED', opacity=0.8, name='Players'
))
fig_hist.add_vline(x=30, line_dash='dash', line_color='#F59E0B',
                   annotation_text='Gate 30', annotation_position='top right',
                   annotation_font_color='#F59E0B')
fig_hist.add_vline(x=40, line_dash='dash', line_color='#10B981',
                   annotation_text='Gate 40', annotation_position='top right',
                   annotation_font_color='#10B981')
fig_hist.update_layout(
    **PLOT_BASE, title='Engagement Distribution (First 100 Rounds)',
    xaxis_title='Game Rounds Played', yaxis_title='Number of Players',
    showlegend=False
)

# 3. Retention by group
metrics = ['Day-1', 'Day-7']
fig_ret = go.Figure()
for version in ['gate_30', 'gate_40']:
    row = retention[retention['version'] == version].iloc[0]
    rates = [row['retention_1'], row['retention_7']]
    fig_ret.add_trace(go.Bar(
        x=metrics, y=rates, name=version,
        marker_color=COLOR_MAP[version],
        text=[f'{r:.2f}%' for r in rates],
        textposition='outside'
    ))
fig_ret.update_layout(
    **PLOT_BASE, title='Day-1 and Day-7 Retention Rates by A/B Group',
    barmode='group', yaxis_range=[0, retention[['retention_1','retention_7']].max().max() * 1.2],
    xaxis_title='', yaxis_title='Retention Rate (%)'
)

# 4. Segment bar
fig_seg = go.Figure()
fig_seg.add_trace(go.Bar(
    x=seg_sorted['count'], y=seg_sorted['segment'],
    orientation='h',
    marker_color=[SEGMENT_COLOR[s] for s in seg_sorted['segment']],
    text=[f"{r}%" for r in seg_sorted['share']],
    textposition='outside'
))
fig_seg.update_layout(
    **PLOT_BASE, title='Player Segmentation by Engagement Level',
    showlegend=False, xaxis_title='Number of Players', yaxis_title='',
    xaxis_range=[0, seg_sorted['count'].max() * 1.15]
)

# 5. Retention by segment
fig_seg_ret = make_subplots(rows=1, cols=2,
    subplot_titles=['Day-1 Retention by Segment', 'Day-7 Retention by Segment'])
for version in ['gate_30', 'gate_40']:
    sub = (seg_ret[seg_ret['version'] == version]
           .set_index('segment').reindex(SEGMENT_ORDER).reset_index())
    fig_seg_ret.add_trace(go.Bar(
        name=version, x=sub['segment'], y=sub['retention_1'],
        marker_color=COLOR_MAP[version], showlegend=True,
        text=[f'{v:.1f}%' for v in sub['retention_1']], textposition='outside'
    ), row=1, col=1)
    fig_seg_ret.add_trace(go.Bar(
        name=version, x=sub['segment'], y=sub['retention_7'],
        marker_color=COLOR_MAP[version], showlegend=False,
        text=[f'{v:.1f}%' for v in sub['retention_7']], textposition='outside'
    ), row=1, col=2)
fig_seg_ret.update_layout(
    **PLOT_BASE, barmode='group',
    title_text='Retention Rates by Segment and A/B Group',
    yaxis_title='Retention Rate (%)', yaxis2_title='Retention Rate (%)'
)
fig_seg_ret.update_xaxes(gridcolor=GRID_COLOR)
fig_seg_ret.update_yaxes(gridcolor=GRID_COLOR)

# 6. Bootstrap
ci1_low, ci1_high = boot1['diff'].quantile([0.025, 0.975])
ci7_low, ci7_high = boot7['diff'].quantile([0.025, 0.975])

fig_boot = make_subplots(rows=1, cols=2, subplot_titles=[
    f'Day-1 - P(gate_30 > gate_40): {prob_1d*100:.1f}%',
    f'Day-7 - P(gate_30 > gate_40): {prob_7d*100:.1f}%'
])
for col, (boot_df, ci_l, ci_h) in enumerate(
    [(boot1, ci1_low, ci1_high), (boot7, ci7_low, ci7_high)], start=1
):
    counts_b, bins_b = np.histogram(boot_df['diff'], bins=60)
    bin_c = (bins_b[:-1] + bins_b[1:]) / 2
    fig_boot.add_trace(go.Bar(
        x=bin_c, y=counts_b, marker_color='#7C3AED', opacity=0.75,
        showlegend=False, name='Frequency'
    ), row=1, col=col)
    for xv, label, color in [(0, 'H₀', '#E11D48'), (ci_l, '2.5%', 'gray'), (ci_h, '97.5%', 'gray')]:
        fig_boot.add_vline(x=xv, line_dash='dash' if xv==0 else 'dot',
                           line_color=color, row=1, col=col)
fig_boot.update_layout(
    **PLOT_BASE, title_text='Bootstrap Distribution of Retention Lift',
    xaxis_title='% Lift (gate_30 − gate_40)', xaxis2_title='% Lift (gate_30 − gate_40)'
)
fig_boot.update_xaxes(gridcolor=GRID_COLOR)
fig_boot.update_yaxes(gridcolor=GRID_COLOR)

# 7. CI scatter
fig_ci = make_subplots(rows=1, cols=2, subplot_titles=[
    f'Day-1 Retention  (p = {r1["p"]})',
    f'Day-7 Retention  (p = {r7["p"]})'
])
for col, res in enumerate([r1, r7], start=1):
    for version, rate, ci in [
        ('gate_30', res['rate_30'], res['ci_30']),
        ('gate_40', res['rate_40'], res['ci_40'])
    ]:
        fig_ci.add_trace(go.Scatter(
            x=[version], y=[rate],
            error_y=dict(type='data',
                         array=[ci[1] - rate], arrayminus=[rate - ci[0]],
                         visible=True, thickness=2, width=10),
            mode='markers',
            marker=dict(color=COLOR_MAP[version], size=14),
            name=version, showlegend=(col == 1)
        ), row=1, col=col)
fig_ci.update_layout(
    **PLOT_BASE, title_text='Retention Rates with 95% Confidence Intervals',
    yaxis_title='Retention Rate (%)', yaxis2_title='Retention Rate (%)'
)
fig_ci.update_xaxes(gridcolor=GRID_COLOR)
fig_ci.update_yaxes(gridcolor=GRID_COLOR)

# Helper: card component 
def card(fig, label, span2=False):
    return html.Div(style={
        'background': CARD_BG,
        'border': f'1px solid {CARD_BORDER}',
        'borderRadius': '12px',
        'overflow': 'hidden',
        'gridColumn': '1 / -1' if span2 else 'span 1'
    }, children=[
        html.Div(label, style={
            'padding': '14px 20px 10px',
            'fontSize': '11px', 'fontWeight': '500',
            'color': TEXT_DIM, 'letterSpacing': '0.09em',
            'textTransform': 'uppercase', 'fontFamily': MONO,
            'borderBottom': f'1px solid rgba(124,58,237,0.1)'
        }),
        dcc.Graph(figure=fig, config={'displayModeBar': False},
                  style={'height': '360px'})
    ])

def section_header(number, title, accent):
    return html.Div(style={
        'display': 'flex', 'alignItems': 'center',
        'gap': '14px', 'marginBottom': '28px'
    }, children=[
        html.Div(style={
            'width': '4px', 'height': '30px', 'borderRadius': '2px',
            'background': f'linear-gradient(180deg, {accent}, {accent}88)'
        }),
        html.H2(f'{number} - {title}', style={
            'margin': '0', 'fontSize': '20px',
            'fontWeight': '600', 'color': '#e8e6f0'
        })
    ])

# Layout
app = dash.Dash(__name__, title='Cookie Cats · A/B Analysis')

app.layout = html.Div(style={
    'fontFamily': FONT,
    'backgroundColor': '#0f0f13',
    'color': '#e8e6f0',
    'minHeight': '100vh',
}, children=[

    html.Link(rel='stylesheet',
              href='https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap'),

    # Header 
    html.Div(style={
        'background': 'linear-gradient(135deg, #1a0a2e 0%, #16213e 55%, #0f3460 100%)',
        'padding': '56px 48px 44px',
        'borderBottom': '1px solid rgba(124,58,237,0.25)',
        'position': 'relative', 'overflow': 'hidden'
    }, children=[
        html.Div(style={
            'position': 'absolute', 'top': '-100px', 'right': '-100px',
            'width': '500px', 'height': '500px', 'borderRadius': '50%',
            'background': 'radial-gradient(circle, rgba(124,58,237,0.12) 0%, transparent 65%)',
            'pointerEvents': 'none'
        }),
        html.H1('Cookie Cats - A/B Test Analysis', style={
            'fontSize': 'clamp(26px, 4vw, 40px)', 'fontWeight': '600',
            'margin': '0 0 10px', 'letterSpacing': '-0.5px', 'color': '#ffffff'
        }),
        html.P('Exploratory Data Analysis & Statistical Hypothesis Testing', style={
            'fontSize': '15px', 'color': 'rgba(232,230,240,0.5)',
            'margin': '0 0 36px', 'fontWeight': '300'
        }),
        # KPI chips
        html.Div(style={'display': 'flex', 'gap': '14px', 'flexWrap': 'wrap'}, children=[
            html.Div(style={
                'background': 'rgba(124,58,237,0.15)',
                'border': '1px solid rgba(124,58,237,0.4)',
                'borderRadius': '10px', 'padding': '14px 22px'
            }, children=[
                html.Div(f'{len(df):,}', style={'fontSize': '26px', 'fontWeight': '600', 'color': '#a78bfa'}),
                html.Div('Players', style={'fontSize': '12px', 'color': TEXT_DIM, 'marginTop': '3px'})
            ]),
            html.Div(style={
                'background': 'rgba(245,158,11,0.12)',
                'border': '1px solid rgba(245,158,11,0.35)',
                'borderRadius': '10px', 'padding': '14px 22px'
            }, children=[
                html.Div('2', style={'fontSize': '26px', 'fontWeight': '600', 'color': '#fcd34d'}),
                html.Div('Groups (gate_30 vs gate_40)', style={'fontSize': '12px', 'color': TEXT_DIM, 'marginTop': '3px'})
            ]),
            html.Div(style={
                'background': 'rgba(16,185,129,0.12)',
                'border': '1px solid rgba(16,185,129,0.35)',
                'borderRadius': '10px', 'padding': '14px 22px'
            }, children=[
                html.Div(f'{prob_7d*100:.1f}%', style={'fontSize': '26px', 'fontWeight': '600', 'color': '#6ee7b7'}),
                html.Div('P(gate_30 better) - Day-7', style={'fontSize': '12px', 'color': TEXT_DIM, 'marginTop': '3px'})
            ]),
            html.Div(style={
                'background': 'rgba(239,68,68,0.1)',
                'border': '1px solid rgba(239,68,68,0.3)',
                'borderRadius': '10px', 'padding': '14px 22px'
            }, children=[
                html.Div('Keep gate_30', style={'fontSize': '18px', 'fontWeight': '600', 'color': '#fca5a5'}),
                html.Div('Recommendation', style={'fontSize': '12px', 'color': TEXT_DIM, 'marginTop': '3px'})
            ]),
        ])
    ]),

    # Section 1: EDA
    html.Div(style={'padding': '44px 48px 0'}, children=[
        section_header('01', 'Exploratory Data Analysis', '#7C3AED'),
    ]),
    html.Div(style={
        'padding': '0 48px', 'display': 'grid',
        'gridTemplateColumns': 'repeat(2, 1fr)',
        'gap': '18px', 'marginBottom': '18px'
    }, children=[
        card(fig_balance, 'Group Balance'),
        card(fig_hist,    'Engagement Distribution'),
    ]),
    html.Div(style={'padding': '0 48px', 'marginBottom': '18px'}, children=[
        card(fig_ret, 'Retention Rates by A/B Group', span2=True),
    ]),
    html.Div(style={
        'padding': '0 48px', 'display': 'grid',
        'gridTemplateColumns': 'repeat(2, 1fr)',
        'gap': '18px', 'marginBottom': '48px'
    }, children=[
        card(fig_seg,     'Player Segmentation'),
        card(fig_seg_ret, 'Retention by Segment'),
    ]),

    # Divider
    html.Div(style={
        'margin': '0 48px 44px',
        'borderTop': '1px solid rgba(124,58,237,0.15)'
    }),

    #  Section 2: A/B Testing 
    html.Div(style={'padding': '0 48px'}, children=[
        section_header('02', 'Statistical Hypothesis Testing', '#F59E0B'),
    ]),
    html.Div(style={'padding': '0 48px', 'marginBottom': '18px'}, children=[
        card(fig_boot, 'Bootstrap Simulation - 10,000 Iterations', span2=True),
    ]),
    html.Div(style={'padding': '0 48px', 'marginBottom': '18px'}, children=[
        card(fig_ci, 'Retention Rates with 95% Confidence Intervals (Z-test)', span2=True),
    ]),

    # Results table
    html.Div(style={'padding': '0 48px', 'marginBottom': '60px'}, children=[
        html.Div(style={
            'background': CARD_BG,
            'border': f'1px solid {CARD_BORDER}',
            'borderRadius': '12px', 'overflow': 'hidden'
        }, children=[
            html.Div('Results Summary', style={
                'padding': '16px 22px 12px', 'fontSize': '11px', 'fontWeight': '500',
                'color': TEXT_DIM, 'borderBottom': 'none',
                'letterSpacing': '0.09em', 'textTransform': 'uppercase', 'fontFamily': MONO
            }),
            html.Table(style={
                'width': '100%', 'borderCollapse': 'collapse', 'fontFamily': FONT
            }, children=[
                html.Thead(html.Tr([
                    html.Th(col, style={
                        'padding': '12px 22px', 'textAlign': 'left',
                        'fontSize': '11px', 'fontWeight': '500', 'color': TEXT_DIM,
                        'letterSpacing': '0.08em', 'textTransform': 'uppercase',
                        'borderBottom': f'1px solid rgba(124,58,237,0.12)'
                    }) for col in ['Metric', 'gate_30 (%)', 'gate_40 (%)',
                                   'Bootstrap P(30>40)', 'p-value', "Cohen's h", 'Significant']
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td('Day-1', style={'padding':'14px 22px','fontSize':'14px','color':TEXT_PRIMARY}),
                        html.Td(f'{r1["rate_30"]:.2f}', style={'padding':'14px 22px','fontSize':'14px','color':'#a78bfa'}),
                        html.Td(f'{r1["rate_40"]:.2f}', style={'padding':'14px 22px','fontSize':'14px','color':'#fcd34d'}),
                        html.Td(f'{prob_1d*100:.1f}%', style={'padding':'14px 22px','fontSize':'14px','color':TEXT_PRIMARY}),
                        html.Td(f'{r1["p"]:.6f}', style={'padding':'14px 22px','fontSize':'14px','color':TEXT_PRIMARY,'fontFamily':MONO}),
                        html.Td(f'{h1:.4f}', style={'padding':'14px 22px','fontSize':'14px','color':TEXT_PRIMARY,'fontFamily':MONO}),
                        html.Td('✓ Yes' if r1['sig'] else '✗ No', style={
                            'padding':'14px 22px','fontSize':'14px',
                            'color':'#6ee7b7' if r1['sig'] else '#f87171', 'fontWeight':'500'
                        }),
                    ], style={'borderTop': '1px solid rgba(255,255,255,0.05)'}),
                    html.Tr([
                        html.Td('Day-7', style={'padding':'14px 22px','fontSize':'14px','color':TEXT_PRIMARY}),
                        html.Td(f'{r7["rate_30"]:.2f}', style={'padding':'14px 22px','fontSize':'14px','color':'#a78bfa'}),
                        html.Td(f'{r7["rate_40"]:.2f}', style={'padding':'14px 22px','fontSize':'14px','color':'#fcd34d'}),
                        html.Td(f'{prob_7d*100:.1f}%', style={'padding':'14px 22px','fontSize':'14px','color':TEXT_PRIMARY}),
                        html.Td(f'{r7["p"]:.6f}', style={'padding':'14px 22px','fontSize':'14px','color':TEXT_PRIMARY,'fontFamily':MONO}),
                        html.Td(f'{h7:.4f}', style={'padding':'14px 22px','fontSize':'14px','color':TEXT_PRIMARY,'fontFamily':MONO}),
                        html.Td('✓ Yes' if r7['sig'] else '✗ No', style={
                            'padding':'14px 22px','fontSize':'14px',
                            'color':'#6ee7b7' if r7['sig'] else '#f87171', 'fontWeight':'500'
                        }),
                    ], style={'borderTop': '1px solid rgba(255,255,255,0.05)'}),
                ])
            ])
        ])
    ]),

    # Footer
    html.Div(style={
        'borderTop': '1px solid rgba(124,58,237,0.12)',
        'padding': '24px 48px',
        'display': 'flex', 'justifyContent': 'space-between',
        'alignItems': 'center', 'flexWrap': 'wrap', 'gap': '10px'
    }, children=[
        html.Span('Cookie Cats A/B Analysis · Portfolio Project',
                  style={'fontSize': '12px', 'color': 'rgba(232,230,240,0.25)'}),
        html.Span('Dataset: Kaggle - Mobile Games A/B Testing',
                  style={'fontSize': '12px', 'color': 'rgba(232,230,240,0.25)', 'fontFamily': MONO})
    ])
])

if __name__ == '__main__':
    app.run(debug=True)
