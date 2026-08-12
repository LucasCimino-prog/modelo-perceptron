import numpy as np
import pandas as pd

class Perceptron:
    """
    Implementação do modelo Perceptron de camada única do zero usando NumPy.
    
    Parâmetros:
    -----------
    lr : float
        Taxa de aprendizado (learning rate), controla a intensidade do ajuste dos pesos.
    epochs : int
        Número de passagens completas pelo conjunto de dados de treinamento.
    """
    def __init__(self, lr=0.1, epochs=10):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Treina o Perceptron ajustando os pesos e o bias iterativamente.
        
        Parâmetros:
        -----------
        X : np.ndarray de forma (n_amostras, n_atributos)
            Vetor de características de entrada.
        y : np.ndarray de forma (n_amostras,)
            Rótulos ou saídas desejadas (0 ou 1).
        """
        n_samples, n_features = X.shape
        
        # Inicializa os pesos e o bias com zero
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        print("Iniciando o treinamento do Perceptron...\n")

        for epoch in range(self.epochs):
            errors_in_epoch = 0
            
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                # Erro = Valor real - Valor previsto
                error = target - prediction
                
                # Regra de atualização dos Pesos e Bias: Δw = η * (y - ŷ) * x
                update = self.lr * error
                self.weights += update * xi
                self.bias += update
                
                if error != 0:
                    errors_in_epoch += 1
                    
            print(f"Época {epoch + 1:02d}/{self.epochs:02d} | Erros: {errors_in_epoch} | Pesos: {self.weights} | Bias: {self.bias:.2f}")

    def predict(self, X):
        """
        Calcula a previsão binária para as entradas fornecidas.
        
        Combinação Linear: z = (W · X) + b
        Função de Ativação Degrau: ŷ = 1 se z >= 0, senão 0
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0.0, 1, 0)

def treinar_e_avaliar(csv_file, feature_cols, target_col, nome_desafio, epocas=20):
    print("\n" + "=" * 70)
    print(f"DESAFIO: {nome_desafio}")
    print("=" * 70)
    
    # 1. Carrega os dados
    df = pd.read_csv(csv_file)
    X = df[feature_cols].values
    y = df[target_col].values
    
    # 2. Treina o modelo 
    modelo = Perceptron(lr=0.1, epochs=epocas)
    modelo.fit(X, y)
    
    # 3. Avalia a Acurácia
    previsoes = modelo.predict(X)
    acuracia = np.mean(previsoes == y) * 100
    
    # 4. Exibição dos resultados
    print(f"\n[ RESULTADOS GERAIS ]")
    print(f"Arquivo analisado: {csv_file}")
    print(f"Acurácia final:    {acuracia:.2f}% (após {epocas} épocas)")
    
    print(f"\n[ PESOS APRENDIDOS PELO MODELO ]")
    print("Como cada variável influenciou a decisão (Positivo = Aprova/Sim, Negativo = Reprova/Não):")
    
    # Faz um loop para mostrar x1, x2, x3 dinamicamente
    for i, nome_coluna in enumerate(feature_cols):
        peso = np.round(modelo.weights[i], 4)
        print(f" -> x{i+1} ({nome_coluna}): {peso}")
        
    print(f" -> Bias (Ajuste da reta): {modelo.bias:.2f}")

    print(f"\n[ STATUS ]")
    if acuracia == 100.0:
        print("SUCESSO: O modelo convergiu perfeitamente! Os dados são linearmente separáveis.")
    else:
        print("ATENÇÃO: O modelo não chegou a 100%. Os dados possuem sobreposição (não são 100% linearmente separáveis).")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    # 1) Análise de Risco de Crédito
    treinar_e_avaliar(
        "massa_credito_100.csv", 
        ['score_credito', 'renda_proporcional'], 
        'aprovado', 
        "1) Análise de Risco de Crédito (Aprovação de Empréstimo)"
    )

    # 2) Filtro de E-mails (Spam)
    treinar_e_avaliar(
        "massa_spam_100.csv", 
        ['palavra_oferta', 'palavra_urgente', 'contato_conhecido'], 
        'eh_spam', 
        "2) Filtro de E-mails (Classificação de Spam)"
    )

    # 3) Triagem Médica (Risco de Diabetes)
    treinar_e_avaliar(
        "massa_saude_100.csv", 
        ['glicemia_relativa', 'imc_relativo', 'historico_familiar'], 
        'prioritario', 
        "3) Triagem Médica (Risco de Diabetes)"
    )

    # 4) Controle de Qualidade Industrial
    treinar_e_avaliar(
        "massa_qualidade_100.csv", 
        ['desvio_diametro_mm', 'desvio_peso_g'], 
        'peca_aprovada', 
        "4) Controle de Qualidade Industrial (Inspeção Mecânica)"
    )