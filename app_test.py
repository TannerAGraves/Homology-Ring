from dash import Dash, dcc, html, dash_table, Input, Output, State, ctx
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
import networkx as nx
from matplotlib import colors

from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform


import pipeline.HomologyRing as HomologyRing
import logging
from pipeline.utils import *

blast_DBs = {'PDB':'blast_db/pdb_seqres', 'SPROT':'blast_db/uniprot_sprot.fasta'}

def plotly_3d(rh, MG, G_dict, x_start, x_end):
    """Prepares node and edge data for 3d RIN plot given RIN output from build_hRIN

    Args:
        rh (RingHomology): current instance of pipeline class
        MG (nx.MultiGraph): Multigraph representing hRIN for all interaction
        G_dict (dict of nx.Graph): dictionary with enteries hRIN for a given entry type key

    Returns:
        trace information for plot as dictionary - one trace for each interaction type.
    """
    inter_types = list(G_dict.keys())
    backbone_x = []
    backbone_y = []
    backbone_z = []
    for n_id in range(rh.msa.get_alignment_length()):
        x = MG.nodes[n_id].get('x')
        if x is not None:
            y = MG.nodes[n_id].get('y')
            z = MG.nodes[n_id].get('z')
            backbone_x.append(x)
            backbone_y.append(y)
            backbone_z.append(z)
        interp_node_pos(rh, n_id, MG, G_dict)
    
    backbone_trace = go.Scatter3d(
        x=backbone_x, y=backbone_y, z=backbone_z,
        line=dict(width=0.5, color='#000'),
        hoverinfo='none',
        mode='lines',
        opacity=0.5
    )
    node_trace_dict = {}
    edge_trace_dict = {}
    for inter_type in inter_types:
        G = G_dict[inter_type]

        present_nodes = set()
        edge_x = []
        edge_y = []
        edge_z = []
        for edge in G.edges():
            if not ((x_start <= edge[0] <= x_end) or (x_start <= edge[1] <= x_end)):
                continue
            x0 = G.nodes[edge[0]].get('x', 0.0)
            y0 = G.nodes[edge[0]].get('y', 0.0)
            z0 = G.nodes[edge[0]].get('z', 0.0)
            x1 = G.nodes[edge[1]].get('x', 0.0)
            y1 = G.nodes[edge[1]].get('y', 0.0)
            z1 = G.nodes[edge[1]].get('z', 0.0)
            present_nodes.update({edge[0], edge[1]})
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])

        edge_trace_dict[inter_type] = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=0.7, color=colors.to_hex(contact_colors[inter_type])),
            hoverinfo='none',
            mode='lines'
        )

        node_x = []
        node_y = []
        node_z = []
        node_text = []
        node_colors = []
        customdata = []
        for n_id in present_nodes:
            n_data = G.nodes[n_id]
            x = n_data.get('x', np.random.randn())
            y = n_data.get('y', np.random.randn())
            z = n_data.get('z', np.random.randn())
            color = n_data.get('color', '#000000') # black to see if any node color data is missing
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            node_colors.append(color)
            cur_node_text = f"{n_data.get('node_type', '')} {str(n_id)}"
            node_name = n_data.get('identifier')
            if node_name is not None:
                cur_node_text += f'\n{node_name}'
            node_text.append(cur_node_text)
            custom_node_dat = {
                'n_id': n_id,
                'type': n_data.get('node_type'),
                'consensus': n_data.get('consensus', "UNK"),
                'db_code': n_data.get('db_code', ''), # used for getting gene chains are transcribed from
                'identifier': n_data.get('identifier', ''), # name assigned to node, used to get UNP ACC
                'pdbx_description': n_data.get('pdbx_description')
            }
            customdata.append(custom_node_dat)

        node_trace_dict[inter_type] = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            marker=dict(size=2, color=node_colors), # tab:blue
            text=node_text,
            hoverinfo='text',
            customdata=customdata
        )
    return node_trace_dict, edge_trace_dict, backbone_trace

def build_contact_plot(rh, inter_class, selected_inters, trim=False, out_type = 'CLUSTAL'):
        if isinstance(selected_inters, str):
            selected_inters = [selected_inters]
            
        sele_l, sele_r, _ = trim_msa(rh.msa, trim)
        sele_r = min(sele_r, rh.msa.get_alignment_length() - 1)

        if out_type == 'CLUSTAL':
            # create the color values for the backgorund of the MSA plot.
            # same for all plots so done once.
            if rh.contact_img_base is None:
                rh.contact_img_base = rh.build_contact_img_base()
            contact_img = rh.contact_img_base.copy()[:, sele_l:sele_r+1]
        elif out_type == 'cmap':
            if rh.contact_image_base_cmap is None:
                rh.contact_image_base_cmap = rh.build_contact_img_base(out_type = 'cmap')
            contact_img = rh.contact_image_base_cmap.copy()[:, sele_l:sele_r+1]
        else:
            contact_img = np.zeros((len(rh.msa),sele_r - sele_l + 1))

        # contact_img_dict = {}
        # h_data_list = []
        # # Theere is one subplot for each interaction type
        # # init the dictionary where keys are the interaction type and values image array
        # for contact_type in rh._interaction_types:
        #     contact_img_dict[contact_type] = rh.contact_img_base.copy()

        

        # map the structure identifiers to the row index in the image
        acc_msa_row_dict = {seq_rec.id: index for index, seq_rec in enumerate(rh.msa)}

        # # filter contact_df according to desired inter_participants
        filtered_contact_df = rh.filter_contact_df(inter_class=inter_class, use_entity_contact_df = True)#self.contact_df[sele]

        for i, r in filtered_contact_df.iterrows():
            acc = r['acc']
            acc_chn = r['acc_chn']
            msa_row = acc_msa_row_dict[acc_chn]
            inter = r['inter']
            msaCol1 = r['msaCol1']
            msaCol2 = r['msaCol2']
            
            if inter not in selected_inters:
                continue

            alpha = 0.00
            if not pd.isna(msaCol1):
                msaCol1 = int(msaCol1)
                if (sele_l <= msaCol1 <= sele_r):
                    res1 = three_to_one(r['NodeId1'].split(':')[-1])
                    if out_type == 'CLUSTAL':
                        entry = np.array(make_darker(hex_to_rgb(clustal_colors[res1]), alpha)) / 255.0
                    elif out_type == 'cmap':
                        entry = res_color_idx_dict[res1]
                    else: # binary
                        entry = 1
                    contact_img[msa_row, msaCol1 - sele_l] = entry
                    # h_data_list.append(
                    #     'row': msaRow,
                    #     'col': msaCol1,
                    #     'acc_chn': acc_chn,
                    #     'other_nid': 
                    # )
                
            if not pd.isna(msaCol2):
                msaCol2 = int(msaCol2)
                if (sele_l <= msaCol2 <= sele_r):
                    res2 = three_to_one(r['NodeId2'].split(':')[-1])
                    if out_type == 'CLUSTAL':
                        entry = np.array(make_darker(hex_to_rgb(clustal_colors[res2]), alpha)) / 255.0
                    elif out_type == 'cmap':
                        entry = res_color_idx_dict[res2]
                    else: # binary contact plots
                        entry = 1
                    contact_img[msa_row, msaCol2 - sele_l] = entry
        return contact_img

app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        dcc.Input(
            id='input-acc',
            placeholder='Input UNP or PDB ID',
            style={
                'height': '40px', 
                'borderRadius': '5px 0 0 5px', 
                'borderRight': 'none', 
                'boxSizing': 'border-box'
            }
        ),
        dcc.Upload(
            id='upload-cif',
            children=html.Div([
                html.A('Upload CIF File')
            ], style={
                'height': '40px', 
                'lineHeight': '40px', 
                'boxSizing': 'border-box',
                'borderRadius': '0'
            }),
            style={
                'width': '150px',
                'height': '40px',
                'lineHeight': '40px',
                'borderWidth': '1px',
                'borderStyle': 'solid',
                'borderRadius': '0 5px 5px 0',
                'textAlign': 'center',
                'boxSizing': 'border-box',
                'marginLeft': '-1px'  
            }
        ),
        html.Span(
            'Query Chain:', 
            style={'marginLeft': '10px', 'lineHeight': '40px'} 
        ),
        dcc.Dropdown(
            id='dropdown-chain-id-source',
            options=[
                {'label': 'LABEL', 'value': 'label_asym_id'},
                {'label': 'AUTH', 'value': 'auth_asym_id'}
            ],
            value='label_asym_id',
            clearable=False,
            style={
                'height': '40px',
                'borderRadius': '5px',
                'boxSizing': 'border-box',
                'marginLeft': '10px', 
                'width': '90px'
            }
        ),
        dcc.Input(
            id='input-chain',
            placeholder='Chain ID',
            style={
                'height': '40px',
                'borderRadius': '5px',
                'boxSizing': 'border-box',
                'marginLeft': '10px'
            }
        ),
        html.Button(
            id='button-run',
            children='Run',
            style={
                'height': '40px', 
                'width': '70px',
                'marginLeft': '20px', 
                'borderRadius': '5px', 
                'boxSizing': 'border-box'
            }
        )
    ], style={
        'display': 'flex', 
        'flexDirection': 'row', 
        'alignItems': 'center'
        #'gap': '10px'
    }),
    html.Div(
        [dcc.Graph(id='3d-graph')],
        style={'width': '75%', 'display': 'inline-block', 'verticalAlign': 'top'}
    ),
    html.Div([
        html.Div(id='selected-obj', style={'whiteSpace': 'pre-wrap'}),
        dash_table.DataTable(
        id='datatable-sele',
        style_table={
            'maxHeight': '300px', 
            'overflowY': 'scroll'
            },
        fixed_rows={'headers': True}
        ),
        dcc.Checklist(
            id='check-group-obs',
            options=[{'label': 'Group Observations', 'value': 'group'}],
            value=['group']
        ),
        dcc.Store(id='stored-selection', data={})
    ], style = {'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    html.Div([
        dcc.Checklist(
            id='show-backbone',
            options=[{'label': 'Show Query Backbone', 'value': 'show'}],
            value=['show']
        ),
        dcc.Dropdown(
            id='dropdown-interaction-class',
            options=[
                {'label': 'Intra-Chain', 'value': 'intra'},
                {'label': 'Inter-Chain', 'value': 'inter'},
                {'label': 'Ligand', 'value': 'LIG'},
                {'label': 'All', 'value': 'all'}
            ],
            value='intra',  # Default value
            clearable=False
        ),
        dash_table.DataTable(
            id='datatable-contact-type',
            columns=[
                {'name': 'Contact Type', 'id': 'contact_type'},
                {'name': 'Count', 'id': 'count'}
            ],
            row_selectable="multi",
            selected_rows = [0]
        ),
        # dcc.Dropdown(
        #     id='dropdown-conservation-normalization',
        #     options=[
        #         {'label': 'Num. Homologs', 'value': 'strong'},
        #         {'label': 'Possible Interactions', 'value': 'possible'}
        #     ],
        #     value='possible',  # Default value
        #     clearable=False,
        #     placeholder="Select Conservation Normalization method",
        #     style={'margin-top': '10px'}  # Optional: Adds some spacing between elements
        # ),
        html.Div([
            html.Label('Normalization method:', style={'display': 'inline-block', 'margin-right': '10px'}),  # Label before the dropdown
            dcc.Dropdown(
                id='dropdown-conservation-normalization',
                options=[
                    {'label': 'Num. Homologs', 'value': 'strong'},
                    {'label': 'Possible Interactions', 'value': 'possible'}
                ],
                value='possible',  # Default value
                clearable=False,
                placeholder="Select Conservation Normalization method",
                style={'margin-top': '10px'}  # Optional: Adds some spacing between elements
            )
        ]),
        dcc.Store(id='stored-selected-inters')], 
        style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top'}
    ),
    html.Div(
        [dcc.Graph(
            id='image-contact-prob')],
        style={'width': 'auto', 'display': 'inline-block', 'verticalAlign': 'top'}
    ),
    html.Div([
        html.Div('Region Filter:'),
        dcc.Dropdown(
            id='dropdown-region-sele-method',
            options=[
                {'label': 'All Columns', 'value':'all'},
                {'label': 'By Occupancy', 'value':'occupancy'},
                {'label': 'Specify Region', 'value':'bounds'}
            ],
            value='all',
            clearable=False,
            style={
                'width':'140px'
            }
        ),
        dcc.Store(id='stored-filter-bounds'),
        html.Div([
            dcc.Input(
                id='input-region-occupancy-thresh',
                placeholder='Occupancy Threshold',
                type='number',
                min=0.0,
                max=1.0,
                step=0.1,
                style={'display': 'none'}  # Initially hidden
            ),
            html.Div([
                dcc.Input(
                    id='input-region-start',
                    placeholder='Start',
                    type='number',
                    style={'marginRight': '10px'},
                    min=0
                ),
                dcc.Input(
                    id='input-region-end',
                    placeholder='End',
                    type='number',
                    min=0
                )
            ], id='bounds-inputs', style={'display': 'none', 'flexDirection': 'row', 'alignItems': 'center'})  # Initially hidden
        ], id='container-region-input'),
        dcc.Checklist(
            id='check-group-sequences',
            options=[{'label': 'Group by Contact Similarity', 'value': 'group'}]
        ),
        dcc.Checklist(
            id='check-similarity-calc-region',
            options=[{'label': 'Calculate Simlarity only on Filtered Region', 'value': 'region'}]
        )
    ], style={'display': 'flex', 'flexDirection': 'row', 'alignItems': 'center', 'gap': '10px'}),
    html.Div(
        [
            dcc.Graph(id='image-MSA-dendro')
        ],
        style={'width': 'auto', 'display': 'inline-block', 'verticalAlign': 'top'}
    ),
    html.Div([
        html.Button(
            id='button-download-nodes',
            children='Download Nodes',
            style={
                'height': '40px', 
                'width': '70px',
                'marginLeft': '20px', 
                'borderRadius': '5px', 
                'boxSizing': 'border-box'
            }
        ),
        html.Button(
            id='button-download-edges',
            children='Download Edges',
            style={
                'height': '40px', 
                'width': '70px',
                'marginLeft': '20px', 
                'borderRadius': '5px', 
                'boxSizing': 'border-box'
            }
        ),
    ], style={
        'display': 'flex', 
        'flexDirection': 'row', 
        'alignItems': 'center'
        #'gap': '10px'
    }),
    dcc.Store(id='stored-data'),
    dcc.Store(id='stored-rowcol-nid-map')
])

@app.callback(
    [Output('input-region-occupancy-thresh', 'style'),
     Output('bounds-inputs', 'style')],
    Input('dropdown-region-sele-method', 'value')
)
def layout_region_input(region_choice):
    if region_choice == 'occupancy':
        return {'display': 'block'}, {'display': 'none'}
    elif region_choice == 'bounds':
        return {'display': 'none'}, {'display': 'flex'}
    return {'display': 'none'}, {'display': 'none'}

@app.callback(
    Output('stored-filter-bounds', 'data'),
    [Input('dropdown-region-sele-method', 'value'),
     Input('input-region-occupancy-thresh', 'value'),
     Input('input-region-start', 'value'),
     Input('input-region-end', 'value')]
)
def get_trim_bounds(filter_option, occupancy_thresh, x_start, x_end):
    if filter_option == 'all':
        trim_input = False
    elif filter_option == 'occupancy':
        trim_input = occupancy_thresh if occupancy_thresh else 0.0
    elif filter_option == 'bounds':
        l = x_start if x_start else 0
        r = x_end if x_end else rh.msa.get_alignment_length() - 1
        trim_input = (l, r)
    else:
        raise Exception("Invalid filter method.")
    l, r, _ = trim_msa(rh.msa, trim_input)
    return {'start': l, 'end': r}

@app.callback(
    Output('image-MSA-dendro', 'figure'),
    [Input('dropdown-interaction-class', 'value'),
     Input('stored-selected-inters', 'data'),
     Input('check-group-sequences', 'value'),
     Input('check-similarity-calc-region', 'value'),
     Input('stored-filter-bounds', 'data')]
)
def build_grouped_msa(inter_class, selected_inters, group_rows, calc_dist_on_trim, filter_bounds):
    if filter_bounds['end'] <= filter_bounds['start']:
        trim = (filter_bounds['start'], rh.msa.get_alignment_length()) # prevent error img of size 0 when user is still typing region
    else:
        trim = (filter_bounds['start'], filter_bounds['end'])

    # Create dendrogram
    calc_dist_on_trim = [] if calc_dist_on_trim is None else calc_dist_on_trim
    trim_dist_mat = trim if 'region' in calc_dist_on_trim else False
    #trim_dist_mat = False
    #print(calc_dist_on_trim)
    dist_mat = rh.dist_matrix(contact_types=selected_inters, inter_class=inter_class, trim_region = trim_dist_mat)
    ytdist = squareform(dist_mat)
    # dist_mat = rh.dist_matrix(contact_types=contact_type,inter_class = inter_class)#np.random.randn(32,32)
    # ytdist = squareform(dist_mat)
    Z = hierarchy.linkage(ytdist, 'average')
    dendro_side = ff.create_dendrogram(np.zeros_like(dist_mat), distfun=lambda X: dist_mat, linkagefun=lambda X: Z, orientation='right') # this gives clusters consistent with scipy
    dendro_permutation = [int(i) for i in dendro_side['layout']['yaxis']['ticktext']]

    # Adjust xaxis for dendrogram
    for i in range(len(dendro_side['data'])):
        dendro_side['data'][i]['xaxis'] = 'x2'

    # Create heatmap
    heat_data = build_contact_plot(rh, inter_class=inter_class, selected_inters=selected_inters, out_type='cmap', trim=trim)
    row_permuted_heat_data = [heat_data[i] for i in dendro_permutation]
    accs = [rec.id for rec in rh.msa]
    row_permuted_accs = [accs[i] for i in dendro_permutation]

    # dendro_side.update_yaxes(
    #     #tickvals=dendro_side['layout']['yaxis']['tickvals'],
    #     ticktext=[accs[int(i)] for i in dendro_leaves]
    # )
    heatmap = go.Heatmap(
        #x = dendro_leaves,
        #y = [accs[i] for i in dendro_permutation],
        z = row_permuted_heat_data,#heat_data,
        colorscale = plotly_heatmap_colorscale,
        zmin=0,
        zmax = list(res_color_idx_dict.values())[-1],
        showscale = False
    )

    # Update x axis for heatmap
    #heatmap['x'] = dendro_side['layout']['yaxis']['tickvals']
    heatmap['y'] = dendro_side['layout']['yaxis']['tickvals'] # updating the vals is needed for plot alignment

    # Add heatmap to dendrogram figure
    dendro_side.add_trace(heatmap)

    dendro_side.update_yaxes(
        tickvals=dendro_side['layout']['yaxis']['tickvals'],
        ticktext=row_permuted_accs
    )

    # Update layout
    dendro_side.update_layout(
        width=1200,
        height=800,
        showlegend=False,
        hovermode='closest'
    )

    # Adjust axis for heatmap and dendrogram
    dendro_side.update_layout(
        xaxis={
            'domain': [0.22, 1],
            'mirror': False,
            'showgrid': False,
            'showline': False,
            'zeroline': False,
            'ticks': ""
        },
        xaxis2={
            'domain': [0, 0.15],
            'mirror': False,
            'showgrid': False,
            'showline': False,
            'zeroline': False,
            'showticklabels': False,
            'ticks': ""
        },
        yaxis={
            'domain': [0, 1],
            'mirror': False,
            'showgrid': False,
            'showline': False,
            'zeroline': False,
            'showticklabels': True,
            'ticks': ""#,
            # 'tickvals': dendro_side['layout']['yaxis']['tickvals'],
            # 'ticktext': [accs[int(i)] for i in dendro_leaves]
        }
    )

    return dendro_side

# 

@app.callback(
    [Output('stored-data', 'data'),
     Output('datatable-contact-type', 'data'),
     Output('datatable-contact-type', 'selected_rows'),
     Output('stored-selected-inters', 'data')],
    [Input('dropdown-interaction-class', 'value'),
     Input('datatable-contact-type', 'selected_rows'),
     Input('stored-filter-bounds', 'data') # TODO: add normalization dropdown
     ]
)
def rebuild_graphs(inter_class, selected_rows, filter_bounds,):
    contact_prob_dict, MG, G_dict = rh.create_hRIN(inter_class=inter_class, set_attr=False, normalize=False)
    x_start, x_end = filter_bounds['start'], filter_bounds['end']
    node_trace_dict, edge_trace_dict, backbone_trace = plotly_3d(rh, MG, G_dict, x_start, x_end)
    contact_type_data = [{'contact_type':k ,'count':len(v.edges)} for k,v in G_dict.items()]
    contact_type_data = sorted(contact_type_data, key = lambda x: contact_order[x['contact_type']]) # give consistent order to contact types
    node_type_dict = {n_id: data['node_type'] for n_id, data in MG.nodes(data=True)}

    def serialize_dok(dok_mat, inter_type = None):
        inter_type = 'data' if inter_type is None else inter_type
        coo_mat = dok_mat.tocoo()
        coo_dict = {
            inter_type: coo_mat.data,
            'row': coo_mat.row,
            'col': coo_mat.col
        }
        return coo_dict
    
    # Convert traces to dictionaries to store them in dcc.Store
    stored_data = {
        'MG': nx.node_link_data(MG),
        'G_dict': {k: nx.node_link_data(v) for k, v in G_dict.items()},
        'node_trace_dict': {k: v.to_plotly_json() for k,v in node_trace_dict.items()},
        'edge_trace_dict': {k: v.to_plotly_json() for k,v in edge_trace_dict.items()},
        'backbone_trace': backbone_trace.to_plotly_json(),
        'contact_type_sele': contact_type_data,
        'contact_count_dict': {k: serialize_dok(v, k) for k,v in contact_prob_dict.items()},
        'node_type_dict': node_type_dict
    }
    
    selected_rows = [val for val in selected_rows if val < len(contact_type_data)] # prevent index out of bounds error when changing contact class
    stored_selected_inters = [inter_data['contact_type'] for i, inter_data in enumerate(contact_type_data) if i in selected_rows]
    return stored_data, contact_type_data, selected_rows, stored_selected_inters

@app.callback(
    Output('3d-graph', 'figure'),
    [Input('show-backbone', 'value'),
     Input('stored-data', 'data'),
     Input('stored-selected-inters', 'data')]
)
def update_3d_graph(show_backbone, stored_data, selected_inters):
    # Reconstruct traces from stored data
    backbone_trace = go.Scatter3d(**stored_data['backbone_trace'])
    if 'show' in show_backbone:
        data = [backbone_trace]
    else:
        data = []
    for contact_type in selected_inters:
        node_trace = go.Scatter3d(**stored_data['node_trace_dict'][contact_type])
        edge_trace = go.Scatter3d(**stored_data['edge_trace_dict'][contact_type])
        data.append(node_trace)
        data.append(edge_trace)

    struct_fig = go.Figure(data=data,
                           layout=go.Layout(
                               title='Homology Residue Interaction Network',
                               titlefont_size=16,
                               showlegend=False,
                               hovermode='closest',
                               margin=dict(b=10, l=20, r=20, t=0),
                               xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                           ))
    return struct_fig

@app.callback(
    [Output('image-contact-prob', 'figure'),
     Output('stored-rowcol-nid-map', 'data')],
    [Input('stored-data', 'data'),
     Input('stored-selected-inters', 'data'),
     Input('stored-filter-bounds', 'data'),
     Input('dropdown-conservation-normalization', 'value')]
)
def update_contact_prob_plot(stored_data, selected_inters, region_bounds, normalization_method):
    #normalization_method = 'possible' # weak is number of homologs, possible is when a contact would be permitted by both residues being non-gaps
    msa_len = rh.msa.get_alignment_length()
    num_seqs = len(rh.msa)
    start_x, end_x = region_bounds['start'], region_bounds['end']
    contact_count_dict = stored_data['contact_count_dict']
    sparse_rows = set()
    img_dict = {}
    for contact_type in selected_inters:
        contact_count_inter = contact_count_dict[contact_type]
        for i, j, count in zip(contact_count_inter['row'], contact_count_inter['col'], contact_count_inter[contact_type]):
            if not ((start_x <= i <= end_x) or (start_x <= j <= end_x)):
                continue
            if (i, j) not in img_dict:
                img_dict[(i, j)] = {}
            sparse_rows.add(i)
            img_dict[i, j][contact_type] = count
            img_dict[i, j]['total_sele'] = img_dict[i, j].get('total_sele', 0) + count

    img = np.zeros((len(sparse_rows), len(sparse_rows)))
    sorted_sparse_node_ids = sorted(sparse_rows)
    node_id_sparse_idx_map = {n_id: i for i, n_id in enumerate(sorted_sparse_node_ids)}
    sparse_idx_node_id_map = {v: k for k, v in node_id_sparse_idx_map.items()}
    # create a dictionary will map the non-res nodes present in a given structure
    # keys: structure acc (PDB/UNP), # values: list of node ids
    # this is used to caclulate the normalizing factor for non-residue nodes by considering interactions to be 'possible' if entity is in structure
    EIDN_struct_presence_dict = rh.entity_df.groupby('src_acc')['node_id'].apply(lambda x: list(x.unique())).to_dict()

    hover_text = []
    for i, row in enumerate(img):
        row_text = []
        u = sparse_idx_node_id_map[i]
        node_type_u = stored_data['node_type_dict'][str(u)]
        # u_nongap = [res != '-' for res in rh.msa[:,u]] if node_type_u == 'RES' else [True] * num_seqs
        u_nongap = [res != '-' for res in rh.msa[:,u]] if node_type_u == 'RES' else [u in EIDN_struct_presence_dict[rec.id.split('_')[0]] for rec in rh.msa]
        for j, value in enumerate(row):
            v = sparse_idx_node_id_map[j]
            edge_dict = img_dict.get((u, v), {}) # dictionary containing the information corresponding to the current edge (pixel in plot)
            node_text = ""
            node_type_v = stored_data['node_type_dict'][str(v)]
            #v_nongap = [res != '-' for res in rh.msa[:,v]] if node_type_v == 'RES' else [True] * num_seqs
            v_nongap = [res != '-' for res in rh.msa[:,v]] if node_type_v == 'RES' else [v in EIDN_struct_presence_dict[rec.id.split('_')[0]] for rec in rh.msa]
            total_sele = edge_dict.get('total_sele', 0)

            node_text += f"u: {u} {node_type_u}"
            if node_type_u != 'RES':
                node_text += f"<br>{rh.identifier_dict[u - msa_len]:.16}"
            node_text += f"<br>v: {v} {node_type_v}"
            if node_type_v != 'RES':
                node_text += f"<br>{rh.identifier_dict[v - msa_len]:.16}"
            node_text += f"<br>Total Selected: {total_sele}"
            for k, count in img_dict.get((u, v), {}).items():
                if k != 'total_sele': # add text indicating how many contacts of each type are observed
                    node_text += f"<br>{k} Count: {count}"
            # normalized probabilities
            if normalization_method == 'strong':
                normalizer = num_seqs
            elif normalization_method == 'possible':
                normalizer = sum(r and s for r, s in zip(u_nongap, v_nongap))
                normalizer = normalizer if normalizer else 1 # Avoide divide by zero. Can happen that MSA permits no possible contacts between given node pair (no mutual non-gapped rows)
            normalizer *= len(selected_inters) # considering two interactions doubles the possible edge observations
            prob = total_sele / normalizer
            node_text += f"<br>Prob: {prob:.3f}"
            row_text.append(node_text)
            img[i, j] = prob # total_sele
        hover_text.append(row_text)

    prob_fig = go.Figure(data=go.Heatmap(
        z=img,
        colorscale='Hot',
        text=hover_text,
        hoverinfo='text'
    ))
    prob_fig.update_layout(
        yaxis=dict(scaleanchor="x", scaleratio=1, autorange='reversed'),
        xaxis=dict(constrain='domain'),
        margin=dict(l=10, r=10, t=10, b=10)
    )

    return prob_fig, sparse_idx_node_id_map


def build_node_adj_table(node_adj, group_observations, selected_inters):
    node_adj_table = []
    # iterate over adjacent nodes
    for adj_n_id, data in dict(node_adj).items():
        # iterate over multi-edges had with one neighbor
        for edge_num, edge_data in data.items():
            row_dict = {}
            inter_type = edge_data['inter']
            if inter_type not in selected_inters:
                continue
            if group_observations:
                row_dict.update({
                    'Partner ID': adj_n_id,
                    'Type': inter_type,
                    'Observations': edge_data['count']
                })
                node_adj_table.append(row_dict)
            else:
                for partner in edge_data['observations']:
                    row_dict.update({
                        'Partner ID': adj_n_id,
                        'Type': inter_type,
                        'Observations': partner
                    })
                    node_adj_table.append(row_dict)
    return node_adj_table

@app.callback(
    Output('stored-selection', 'data'),
    [Input('3d-graph', 'clickData'),
     Input('image-contact-prob', 'clickData')],
    [State('stored-selection', 'data'),
     State('stored-rowcol-nid-map', 'data')]
)
def update_selection(node_click_data, edge_click_data, stored_selection, nid_map):
    if not ctx.triggered:
        return {}
    else:
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if triggered_id == '3d-graph' and node_click_data: # click event was on a node (ignore clicks on graph not assoicated w/ node)
            point = node_click_data['points'][0]
            node_info = point.get('customdata')
            if node_info is None:
                return stored_selection # graph click event was not on a node - don't change the store and ignore event
            n_id = node_info['n_id']
            return {'sele_type': 'node', 'n_id': n_id, 'node_type': node_info['type']}
        elif triggered_id == 'image-contact-prob' and edge_click_data:
            point = edge_click_data['points'][0]
            u_id, v_id = nid_map[str(point['x'])], nid_map[str(point['y'])]
            return {'sele_type': 'edge', 'u_id': u_id, 'v_id': v_id}
        else:
            return {}


@app.callback(
    [Output('selected-obj', 'children'),
     Output('datatable-sele', 'data')],
    [Input('stored-selection', 'data'),
     Input('stored-data', 'data'),
     Input('check-group-obs', 'value')],
     State('stored-selected-inters', 'data')
)
def display_click_data(sele_data, stored_data, group_obs, selected_inters):
    #ctx = callback_context
    init_text = "Click on a node / edge to see details here."
    init_table = [{'Selection Info': "Select Node from Structure Plot or Edge from Heatmap to display Summary Data."}]
    MG = nx.node_link_graph(stored_data['MG'])
    sele_type = sele_data.get('sele_type')
    if sele_type == 'node':
        n_id = sele_data['n_id']
        node_type = sele_data['node_type']
        node_adj = MG.adj[n_id]
        node_data = MG.nodes[n_id]
        sele_text = f"Selected Node: {n_id}"
        if node_type == 'RES':
            sele_text += "\nType: Hom. Residue"
            res_consensus = node_data.get('consensus', "Calculate Me!")
            sele_text += f"\nConsensus: {res_consensus}"
        elif node_type == 'CHAIN':
            sele_text += "\nType: Peptide Chain"
            node_desc = node_data.get('pdbx_description', None)
            if node_desc:
                sele_text += f"\n{node_desc}"
            sele_text += f"\n{'id'}: {node_data['identifier']}: {node_data['db_code']}"
        elif node_type == 'LIG':
            sele_text += "\nType: Ligand"
        else: # debug, shouldn't happen
            sele_text += '\nUNKNOWN NODE TYPE'

        node_adj_table = build_node_adj_table(node_adj, group_obs, selected_inters)
        node_adj_table = sorted(node_adj_table, key = lambda x: x['Partner ID'])
        return sele_text, node_adj_table
    
    elif sele_type == 'edge':
        u_id, v_id = sele_data['u_id'], sele_data['v_id']
        sele_text = f"Selected Edge: \nu {u_id} {stored_data['node_type_dict'][str(u_id)]}\nv {v_id} {stored_data['node_type_dict'][str(v_id)]}"
        selected_edges = MG[u_id].get(v_id, {}) # safe if graph has no edges between given nodes
        edge_table = []
        for edge_num, edge_data in selected_edges.items():
            inter_type = edge_data['inter']
            if inter_type not in selected_inters:
                continue
            if group_obs:
                edge_table.append({
                    'Type': inter_type,
                    'Observations': edge_data['count']
                })
            else:
                for obs in edge_data['observations']:
                    edge_table.append({
                        'Type': inter_type,
                        'Observations': obs
                    })
        return sele_text, edge_table
    else:
        return init_text, init_table

# Disable Runbutton unless given valid query
@app.callback(
    Output('button-run', 'disabled'),
    [
        Input('input-acc', 'value'),
        Input('upload-cif', 'contents'),
        Input('input-chain', 'value')
    ]
)
def toggle_button(input_acc, uploaded_file, input_chain):
    valid_chain = input_chain and re.fullmatch(r'[A-Z]{1,2}', input_chain)

    if (input_acc or uploaded_file) and valid_chain:
        return False  # Enable the button
    else:
        return True  # Disable the button

# Run the pipeline for given query
@app.callback(
    #[Output()]
    [Input('button-run', 'n_clicks')],
    [State('input-acc', 'value'),
     State('upload-cif', 'contents'),
     State('input-chain', 'value'),
     State('dropdown-chain-id-source', 'value')]
)
def pipeline_run(n_clicks, input_acc, cif_upload, chain_id, chain_source):

    rh = HomologyRing.HomologyRing('query/1LM8.cif', blast_DBs['PDB'], struct_source='PDB', log_level=logging.DEBUG, uses_PDB_ids=True)
    rh.build(chain_id,'1', force_download=False, force_RING=False, max_results=32, use_label_asym_id=(chain_source=='LABEL'))

if __name__ == '__main__':
    # normal test 8cmz
    # alt test query/2v8l.cif
    # query/1LM8.cif
    #rh = RingHomology.RingHomology('','')
    
    rh = HomologyRing.HomologyRing('query/1LM8.cif', blast_DBs['PDB'], struct_source='PDB', log_level=logging.DEBUG, uses_PDB_ids=True)
    rh.build('C','1', force_download=False, force_RING=False, max_results=32, use_label_asym_id=False, prob_normalization='possible')

    ### DDX USER DEFINED FAMILY
    # rh = RingHomology.RingHomology('DDX_human_pdb/DDX2B_Q14240/3bor.cif', blast_DBs['PDB'], struct_source='PDB', log_level=logging.DEBUG, uses_PDB_ids=True)
    # rh.verbose = True
    # family_dir = 'DDX_human_pdb'
    # tst = rh.build_fromFamily(family_dir, 'A','1', force_download=False, force_RING=False, max_results=64, use_label_asym_id=False)

    app.run_server(debug=True)

