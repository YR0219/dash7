#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# dashboard.py

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, ALL
from flask import Flask, jsonify, request
import pandas as pd
import plotly.graph_objs as go
import joblib
import shap
import numpy as np

# Charger le modèle, le scaler et le seuil optimal
model = joblib.load('best_lgbm_model.pkl')
scaler = joblib.load('scaler.pkl')
best_threshold = joblib.load('best_threshold.pkl')

# Charger les données des clients
clients_data = pd.read_csv('X_test_select.csv')

# Initialisation de l'expliqueur SHAP
explainer = shap.TreeExplainer(model)

# Initialisation de l'application Flask
server = Flask(__name__)

# Initialisation de l'application Dash
app = dash.Dash(__name__, server=server)

# Styles pour le bouton et la prédiction
button_style = {
    'background-color': '#4CAF50',
    'color': 'white',
    'height': '50px',
    'width': '100%',
    'margin-top': '20px',
    'margin-bottom': '20px',
    'font-size': '20px',
    'border': 'none',
    'border-radius': '5px',
    'cursor': 'pointer',
    'padding': '14px 28px'
}

prediction_style_default = {
    'margin-top': '20px',
    'font-size': '20px',
    'font-weight': 'bold',
    'text-align': 'center'
}

# Définition de la mise en page de l'application
app.layout = html.Div([
    dcc.Dropdown(
        id='client-dropdown',
        options=[{'label': str(sk_id), 'value': sk_id} for sk_id in clients_data['sk_id_curr']],
        value=clients_data['sk_id_curr'].iloc[0]
    ),
    html.Div(id='client-data-inputs'),
    html.Button('Mettre à jour la prédiction', id='predict-button', n_clicks=0, style=button_style),
    html.Div(id='prediction-output', style=prediction_style_default),
    dcc.Graph(id='score-gauge'),
    # ... [Autres composants de l'interface utilisateur]
])

# Fonction de rappel pour peupler les entrées de données du client en fonction du client sélectionné
@app.callback(
    Output('client-data-inputs', 'children'),
    [Input('client-dropdown', 'value')]
)
def update_client_data_inputs(selected_sk_id):
    selected_client_data = clients_data[clients_data['sk_id_curr'] == selected_sk_id].iloc[0]
    return [
        html.Div([
            html.Label(col),
            dcc.Input(
                value=selected_client_data[col],
                type='number',
                id={'type': 'input-field', 'index': col}
            )
        ]) for col in selected_client_data.index if col != 'sk_id_curr'
    ]

# Fonctions de rappel pour mettre à jour la prédiction et les graphiques
# Fonction de rappel pour mettre à jour la prédiction et tous les graphiques
@app.callback(
    [Output('prediction-output', 'children'),
     Output('score-gauge', 'figure'),
     Output('feature-view-graph', 'figure'),
     Output('client-comparison', 'figure'),
     Output('global-feature-importance', 'figure'),
     Output('local-feature-importance', 'figure')],
    [Input('predict-button', 'n_clicks'),
     Input('feature-view-dropdown', 'value'),
     Input('feature-dropdown', 'value')],
    [State({'type': 'input-field', 'index': ALL}, 'value'),
     State('client-dropdown', 'value')]
)
def update_all_graphs(n_clicks, selected_feature_view, selected_feature_compare, input_values, selected_sk_id):
    ctx = dash.callback_context
    if not ctx.triggered or ctx.triggered[0]['prop_id'].split('.')[0] == 'client-dropdown':
        raise dash.exceptions.PreventUpdate

    # Récupérer et mettre à jour les données du client sélectionné
    selected_client_data = clients_data[clients_data['sk_id_curr'] == selected_sk_id].iloc[0]
    updated_client_data = selected_client_data.copy()
    for col, val in zip(updated_client_data.index, input_values):
        updated_client_data[col] = val

    # Prétraitement et prédiction
    processed_inputs = scaler.transform([updated_client_data.drop('sk_id_curr')])
    score = model.predict_proba(processed_inputs)[0][1]
    prediction_text = f"Le client est {'Difficult' if score > best_threshold else 'No Difficult'} avec un score de {score:.2f}"

    # Création du graphique de la jauge
    gauge_chart = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Score de Risque"},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "lightblue"},
            'steps': [
                {'range': [0, best_threshold], 'color': "lightgreen"},
                {'range': [best_threshold, 1], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': best_threshold
            }
        }
    ))

    # Création du graphique de la caractéristique sélectionnée
    feature_value_graph = go.Figure(data=[
        go.Bar(x=[selected_feature_view], y=[updated_client_data[selected_feature_view]])
    ])
    feature_value_graph.update_layout(title_text=f"Valeur de {selected_feature_view} pour le client sélectionné")

    # Création du graphique de comparaison
    comparison_graph = go.Figure()
    comparison_graph.add_trace(go.Box(y=clients_data[selected_feature_compare], name='Tous les clients'))
    comparison_graph.add_trace(go.Scatter(x=[selected_feature_compare], y=[updated_client_data[selected_feature_compare]], mode='markers', name='Client Sélectionné', marker=dict(color='red')))
    comparison_graph.update_layout(title_text=f"Comparaison de {selected_feature_compare} avec l'ensemble des clients")

    # Création du graphique d'importance des caractéristiques globale
    feature_importances = model.feature_importances_
    features = [col for col in clients_data.columns if col != 'sk_id_curr']
    sorted_indices = np.argsort(feature_importances)[::-1]
    global_importance_graph = go.Figure(data=[
        go.Bar(x=np.array(features)[sorted_indices], y=feature_importances[sorted_indices])
    ])
    global_importance_graph.update_layout(title_text="Importance Globale des Caractéristiques")

    # Création du graphique d'importance des caractéristiques locale
    processed_data = scaler.transform([updated_client_data.drop('sk_id_curr')])
    shap_values = explainer.shap_values(processed_data)
    shap_values = shap_values[0] if isinstance(shap_values, list) else shap_values
    sorted_indices = np.argsort(shap_values[0])[::-1]
    local_importance_graph = go.Figure(data=[
        go.Bar(x=np.array(features)[sorted_indices], y=shap_values[0][sorted_indices])
    ])
    local_importance_graph.update_layout(title_text="Importance Locale des Caractéristiques pour le Client Sélectionné")

    return prediction_text, gauge_chart, feature_value_graph, comparison_graph, global_importance_graph, local_importance_graph


# Définition d'une route API pour obtenir le score d'un client
@server.route('/get_score', methods=['POST'])
def get_score():
    data = request.get_json()
    id_client = data['id_client']
    client_data = clients_data[clients_data['sk_id_curr'] == id_client]
    
    if client_data.empty:
        response = {
            'message': 'Client not found'
        }
        return jsonify(response), 404
    
    # Préparation des données pour la prédiction
    client_data = client_data.drop(columns=['sk_id_curr'])
    client_data_scaled = scaler.transform(client_data)
    
    # Obtention du score
    score = model.predict_proba(client_data_scaled)[0][1]
    
    response = {
        'id_client': int(id_client),
        'score': score
    }
    return jsonify(response), 200

# Exécution de l'application
if __name__ == '__main__':
    app.run_server(debug=True)

