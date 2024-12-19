import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, dash_table
import io
import base64
import plotly.io as pio
import numpy as np
import json
from tempfile import mkdtemp
from dash import dcc
from dash.exceptions import PreventUpdate
import os
from datetime import datetime
# Initialize Dash app with external stylesheets
app = Dash(__name__)

# Create a temporary directory for saving files
TEMP_DIR = mkdtemp()

# Color schemes
COLOR_SCHEMES = {
    "Viridis": px.colors.sequential.Viridis,
    "Plasma": px.colors.sequential.Plasma,
    "Blues": px.colors.sequential.Blues,
    "Reds": px.colors.sequential.Reds,
    "Custom": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEEAD"]
}

# Global variables
dataset = None
original_dataset = None

def parse_contents(contents, filename):
    """Parse uploaded file contents and return DataFrame with success message"""
    if contents is None:
        return None, "No file uploaded"
    
    try:
        # Decode the uploaded content
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Read different file types
        try:
            if 'csv' in filename:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            elif 'xls' in filename:
                df = pd.read_excel(io.BytesIO(decoded))
            elif 'json' in filename:
                df = pd.read_json(io.StringIO(decoded.decode('utf-8')))
            else:
                return None, f'Unsupported file type: {filename}'
            
            return df, f'Successfully loaded {filename} with {len(df)} rows and {len(df.columns)} columns'
        except Exception as e:
            return None, f'Error processing {filename}: {str(e)}'
            
    except Exception as e:
        return None, f'Error reading file: {str(e)}'

# Custom CSS
app.layout = html.Div([
    # Header Section
    html.Div([
        html.H1("📊 Interactive Data Visualization Dashboard", 
                style={"textAlign": "center", "color": "#2C3E50", "marginBottom": "20px"}),
        html.P("Upload your dataset and create beautiful visualizations instantly!", 
               style={"textAlign": "center", "color": "#7F8C8D"})
    ], className="header-section"),

    # Main Content Container
    html.Div([
        # Left Panel - Controls
        html.Div([
            # File Upload Section
            html.Div([
                html.H3("📁 Data Upload", style={"color": "#2C3E50"}),
                dcc.Upload(
                    id="upload-data",
                    children=html.Div([
                        '📎 Drag and Drop or ',
                        html.A('Select a File (CSV, Excel, JSON)')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px 0'
                    },
                    multiple=False
                ),
                # Upload Status Display
                html.Div(id='upload-status', style={'margin': '10px 0', 'color': '#2C3E50'}),
                # Preview of Uploaded Data
                html.Div(id='data-preview-container', style={'margin': '20px 0'})
            ], className="upload-section"),
            # Visualization Controls
            html.Div([
                html.H3("🎨 Visualization Settings", style={"color": "#2C3E50"}),
                
                # Graph Type Selection
                html.Label("📊 Chart Type:"),
                dcc.Dropdown(
                    id="graph-type",
                    options=[
                        {"label": "Bar Chart", "value": "bar"},
                        {"label": "Line Chart", "value": "line"},
                        {"label": "Scatter Plot", "value": "scatter"},
                        {"label": "Histogram", "value": "histogram"},
                        {"label": "Box Plot", "value": "box"},
                        {"label": "Heatmap", "value": "heatmap"},
                        {"label": "Pie Chart", "value": "pie"},
                        {"label": "3D Scatter", "value": "scatter_3d"},
                        {"label": "Bubble Chart", "value": "bubble"},
                        {"label": "Area Chart", "value": "area"}
                    ],
                    value="bar",
                    className="dropdown"
                ),

                # Axis Selection
                html.Label("📏 X-axis:"),
                dcc.Dropdown(id="x-axis-column", className="dropdown"),
                
                html.Label("📏 Y-axis:"),
                dcc.Dropdown(id="y-axis-column", className="dropdown"),
                
                # Color Scheme Selection
                html.Label("🎨 Color Scheme:"),
                dcc.Dropdown(
                    id="color-scheme",
                    options=[{"label": k, "value": k} for k in COLOR_SCHEMES.keys()],
                    value="Viridis",
                    className="dropdown"
                ),

                # Animation Toggle
                html.Label("✨ Animation:"),
                dcc.Checklist(
                    id='animation-toggle',
                    options=[{'label': 'Enable Animation', 'value': 'animate'}],
                    value=[],
                    className="checklist"
                ),
            ], className="visualization-controls"),

            # Advanced Features
            html.Div([
                html.H3("🔧 Advanced Features", style={"color": "#2C3E50"}),
                
                # Data Filtering
                html.Label("🔍 Filter Data:"),
                dcc.Dropdown(id="filter-column", placeholder="Select Column", className="dropdown"),
                dcc.Input(id="filter-value", type="text", placeholder="Filter Value", className="input"),
                html.Button("Apply Filter", id="apply-filter", className="button"),
                html.Button("Reset Filters", id="reset-filter", className="button"),

                # Statistical Analysis
                html.Label("📈 Statistical Analysis:"),
                dcc.Checklist(
                    id='stats-toggle',
                    options=[
                        {'label': 'Show Trend Line', 'value': 'trend'},
                        {'label': 'Show Summary Stats', 'value': 'stats'}
                    ],
                    value=[],
                    className="checklist"
                ),
            ], className="advanced-features"),

        ], className="left-panel"),

        # Right Panel - Visualization
        html.Div([
            # Graph Output
            dcc.Graph(id="graph-output", className="graph-output"),
            
            # Data Preview
            html.Div([
                html.H3("📋 Data Preview", style={"color": "#2C3E50"}),
                html.Div(id="data-preview", className="data-preview")
            ]),

            # Statistics Output
            html.Div(id="stats-output", className="stats-output"),

            # Export Options
            html.Div([
        # Download Format Selection
        dcc.Dropdown(
            id='download-format',
            options=[
                {'label': 'PNG Image', 'value': 'png'},
                {'label': 'JPEG Image', 'value': 'jpeg'},
                {'label': 'SVG Vector', 'value': 'svg'},
                {'label': 'HTML Interactive', 'value': 'html'},
                {'label': 'PDF Document', 'value': 'pdf'}
            ],
            value='png',
            placeholder="Select download format",
            style={'width': '200px', 'margin': '10px 0'}
        ),
        # Download Quality Selection (for raster formats)
        dcc.Dropdown(
            id='download-quality',
            options=[
                {'label': 'Normal Quality', 'value': 1},
                {'label': 'High Quality', 'value': 2},
                {'label': 'Ultra Quality', 'value': 4}
            ],
            value=1,
            placeholder="Select quality",
            style={'width': '200px', 'margin': '10px 0'}
        ),
        html.Button(
            "📥 Download Graph", 
            id="download-button",
            className="button",
            style={
                'backgroundColor': '#4CAF50',
                'color': 'white',
                'padding': '10px 20px',
                'border': 'none',
                'borderRadius': '5px',
                'cursor': 'pointer',
                'margin': '10px 0'
            }
        ),
        html.Div(id="download-status", style={'margin': '10px 0'}),
        dcc.Download(id="download-graph")
    ], className="export-options"),
        ], className="right-panel")
    ], className="main-content"),

], className="dashboard-container")


@app.callback(
    [Output("download-graph", "data"),
     Output("download-status", "children")],
    [Input("download-button", "n_clicks")],
    [State("graph-output", "figure"),
     State("download-format", "value"),
     State("download-quality", "value")],
    prevent_initial_call=True
)
def download_graph(n_clicks, figure, format_type, scale):
    if n_clicks is None or figure is None:
        raise PreventUpdate

    try:
        # Create a unique filename
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"visualization_{timestamp}.{format_type}"
        filepath = os.path.join(TEMP_DIR, filename)

        # Create a Figure object
        fig = go.Figure(figure)

        # Set the figure size and quality
        width = 1200 * scale
        height = 800 * scale

        if format_type in ['png', 'jpeg', 'pdf']:
            # Write the image to bytes buffer instead of file
            img_bytes = fig.to_image(
                format=format_type,
                width=width,
                height=height,
                scale=scale
            )
            
            return (
                dcc.send_bytes(
                    img_bytes,
                    filename
                ),
                html.Div(f"Download successful! Saved as {filename}", 
                        style={'color': 'green'})
            )

        elif format_type == 'svg':
            # For SVG format
            svg_bytes = fig.to_image(format='svg')
            
            return (
                dcc.send_bytes(
                    svg_bytes,
                    filename
                ),
                html.Div(f"Download successful! Saved as {filename}", 
                        style={'color': 'green'})
            )

        elif format_type == 'html':
            # For HTML format (interactive plot)
            html_str = fig.to_html(include_plotlyjs=True)
            
            return (
                dcc.send_string(
                    html_str,
                    filename
                ),
                html.Div("Download successful! Saved as interactive HTML", 
                        style={'color': 'green'})
            )

    except Exception as e:
        return (
            None,
            html.Div([
                html.P("Error during download:", style={'color': 'red'}),
                html.P(str(e), style={'color': 'red'})
            ])
        )

    if n_clicks is None or figure is None:
        raise PreventUpdate

    try:
        # Create a unique filename
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"visualization_{timestamp}.{format_type}"
        filepath = os.path.join(TEMP_DIR, filename)

        # Create a Figure object
        fig = go.Figure(figure)

        # Set the figure size and quality
        width = 1200 * scale
        height = 800 * scale

        if format_type in ['png', 'jpeg', 'pdf']:
            # For raster formats and PDF
            pio.write_image(
                fig,
                filepath,
                format=format_type,
                width=width,
                height=height,
                scale=scale
            )
            with open(filepath, 'rb') as f:
                encoded_image = base64.b64encode(f.read()).decode()
            
            # Clean up the temporary file
            os.remove(filepath)
            
            return (
                dcc.send_data_frame(
                    lambda x: encoded_image,
                    filename,
                    type=f'image/{format_type}'
                ),
                html.Div(f"Download successful! Saved as {filename}", 
                        style={'color': 'green'})
            )

        elif format_type == 'svg':
            # For SVG format
            fig.write_image(filepath, format='svg')
            with open(filepath, 'r') as f:
                svg_data = f.read()
            
            # Clean up the temporary file
            os.remove(filepath)
            
            return (
                dict(
                    content=svg_data,
                    filename=filename,
                    type='image/svg+xml'
                ),
                html.Div(f"Download successful! Saved as {filename}", 
                        style={'color': 'green'})
            )

        elif format_type == 'html':
            # For HTML format (interactive plot)
            html_str = pio.to_html(fig, include_plotlyjs=True)
            
            return (
                dict(
                    content=html_str,
                    filename=f"visualization_{timestamp}.html",
                    type='text/html'
                ),
                html.Div("Download successful! Saved as interactive HTML", 
                        style={'color': 'green'})
            )

    except Exception as e:
        return (
            None,
            html.Div([
                html.P("Error during download:", style={'color': 'red'}),
                html.P(str(e), style={'color': 'red'})
            ])
        )

        # Add callback to control quality dropdown visibility
@app.callback(
    [Output("download-quality", "style"),
     Output("download-quality", "disabled")],
    [Input("download-format", "value")]
)
def toggle_quality_selector(format_type):
    if format_type in ['png', 'jpeg']:
        return {'width': '200px', 'margin': '10px 0'}, False
    return {'width': '200px', 'margin': '10px 0', 'display': 'none'}, True

# Callback for file upload
@app.callback(
    [Output('upload-status', 'children'),
     Output('data-preview-container', 'children'),
     Output('x-axis-column', 'options'),
     Output('y-axis-column', 'options'),
     Output('filter-column', 'options')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_output(contents, filename):
    global dataset, original_dataset
    
    if contents is None:
        return (
            "No file uploaded yet",
            None,
            [], [], []
        )
    
    # Parse the uploaded file
    df, message = parse_contents(contents, filename)
    
    if df is not None:
        dataset = df
        original_dataset = df.copy()
        
        # Create column options for dropdowns
        column_options = [{'label': col, 'value': col} for col in df.columns]
        
        # Create a preview table
        preview_table = dash_table.DataTable(
            data=df.head(5).to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '5px',
                'backgroundColor': 'white',
                'minWidth': '100px'
            },
            style_header={
                'backgroundColor': '#f8f9fa',
                'fontWeight': 'bold'
            }
        )
        
        return (
            html.Div([
                html.P(message, style={'color': 'green'}),
                html.P(f"Columns available: {', '.join(df.columns)}")
            ]),
            html.Div([
                html.H4("Data Preview (First 5 rows)"),
                preview_table
            ]),
            column_options,
            column_options,
            column_options
        )
    else:
        return (
            html.Div(message, style={'color': 'red'}),
            None,
            [], [], []
        )

# Enhanced callback for graph updates
@app.callback(
    Output("graph-output", "figure"),
    [Input("graph-type", "value"),
     Input("x-axis-column", "value"),
     Input("y-axis-column", "value"),
     Input("color-scheme", "value"),
     Input("animation-toggle", "value"),
     Input("stats-toggle", "value")]
)
def update_graph(graph_type, x_column, y_column, color_scheme, animation, stats_options):
    if dataset is None or x_column is None:
        return px.scatter(title="Please upload data and select columns")
    
    try:
        colors = COLOR_SCHEMES[color_scheme]
        
        # Handle different graph types
        if graph_type == "scatter_3d":
            fig = px.scatter_3d(dataset, x=x_column, y=y_column, 
                              z=dataset.select_dtypes(include=['number']).columns[0],
                              color=dataset[x_column],
                              title=f"3D Scatter Plot")
            
        elif graph_type == "heatmap":
            corr_matrix = dataset.select_dtypes(include=['number']).corr()
            fig = px.imshow(corr_matrix, 
                          color_continuous_scale=colors,
                          title="Correlation Heatmap")
            
        elif graph_type == "bubble":
            size_col = dataset.select_dtypes(include=['number']).columns[0]
            fig = px.scatter(dataset, x=x_column, y=y_column,
                           size=size_col, color=x_column,
                           title=f"Bubble Chart")
            
        elif graph_type == "area":
            fig = px.area(dataset, x=x_column, y=y_column,
                         title=f"Area Chart")
            
        else:
            fig = getattr(px, graph_type)(
                dataset, x=x_column, y=y_column,
                color_discrete_sequence=colors,
                title=f"{graph_type.title()} Chart"
            )

        # Add trend line if requested
        if 'trend' in stats_options and graph_type in ['scatter', 'bubble']:
            x_numeric = pd.to_numeric(dataset[x_column], errors='coerce')
            y_numeric = pd.to_numeric(dataset[y_column], errors='coerce')
            
            if not (x_numeric.isna().all() or y_numeric.isna().all()):
                z = np.polyfit(x_numeric.dropna(), y_numeric.dropna(), 1)
                p = np.poly1d(z)
                x_range = np.linspace(x_numeric.min(), x_numeric.max(), 100)
                fig.add_trace(go.Scatter(x=x_range, y=p(x_range),
                                       name='Trend Line',
                                       line=dict(color='red', dash='dash')))

        # Update layout for better appearance
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#2C3E50'),
            title_x=0.5,
            title_font=dict(size=20),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.8)'
            ),
            margin=dict(l=40, r=40, t=40, b=40)
        )

        # Add animations if enabled
        if 'animate' in animation:
            fig.update_layout(
                transition_duration=500,
                transition={'duration': 500}
            )

        return fig
    
    except Exception as e:
        return px.scatter(title=f"Error: {str(e)}")

# Add callbacks for new features
@app.callback(
    Output("stats-output", "children"),
    [Input("stats-toggle", "value"),
     Input("x-axis-column", "value"),
     Input("y-axis-column", "value")]
)
def update_stats(stats_options, x_column, y_column):
    if 'stats' not in stats_options or dataset is None:
        return ""
    
    try:
        stats_html = html.Div([
            html.H4("📊 Statistical Summary"),
            html.Table([
                html.Tr([html.Th("Metric"), html.Th("Value")]),
                html.Tr([html.Td("Mean"), html.Td(f"{dataset[y_column].mean():.2f}")]),
                html.Tr([html.Td("Median"), html.Td(f"{dataset[y_column].median():.2f}")]),
                html.Tr([html.Td("Std Dev"), html.Td(f"{dataset[y_column].std():.2f}")]),
                html.Tr([html.Td("Correlation"), 
                        html.Td(f"{dataset[x_column].corr(dataset[y_column]):.2f}")])
            ])
        ])
        return stats_html
    except:
        return "Unable to calculate statistics for selected columns"

# Add callback for data export
@app.callback(
    Output("download-data", "data"),
    Input("export-data", "n_clicks"),
    prevent_initial_call=True
)
def export_data(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    return dcc.send_data_frame(dataset.to_csv, "exported_data.csv")

if __name__ == "__main__":
    app.run_server(debug=True)
