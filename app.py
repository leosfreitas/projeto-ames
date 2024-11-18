
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the model once when the app starts
try:
    modelo = joblib.load('trained_model.pkl')
    print("Modelo carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    modelo = None

@app.route('/prever', methods=['POST'])
def prever_preco():
    if modelo is None:
        return jsonify({'error': 'Modelo não está disponível.'}), 500
    try:
        dados = request.get_json()
        if not dados:
            return jsonify({'error': 'Nenhum dado fornecido.'}), 400
        df_input = pd.DataFrame([dados])
        preco_predito = modelo.predict(df_input)[0]
        preco_predito_float = float(preco_predito)
        return jsonify({'preco_predito': preco_predito_float})
    except Exception as e:
        print(f"Erro: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Heroku assigns a port, so use it if available
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=False, host='0.0.0.0', port=port)
