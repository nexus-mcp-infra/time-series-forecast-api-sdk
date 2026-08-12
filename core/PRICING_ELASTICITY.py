import math

def demanda(precio):
    """Función de demanda basada en developer tools"""
    return 10000 / (precio + 2)

def elasticidad(precio, volumen):
    """Elasticidad precio-demanda"""
    d_q_d_p = -volumen * (precio + 2)**(-2)
    return d_q_d_p * (precio / volumen)

def revenue_max(prices):
    """Punto de máxima recaudación"""
    max_revenue = 0
    optimal_price = None
    for price in prices:
        volume = demanda(price)
        revenue = price * volume
        if revenue > max_revenue:
            max_revenue = revenue
            optimal_price = price
    return optimal_price, max_revenue

def scenarios():
    """Escenarios de adopción"""
    return [
        {'segmento': 'early_adopter', 'precio': 0.01, 'volumen_mensual': 5000},
        {'segmento': 'mid_market', 'precio': 0.02, 'volumen_mensual': 50000},
        {'segmento': 'enterprise', 'precio': 0.04, 'volumen_mensual': 500000},
    ]

def entropia_condicional(predictive_distribution):
    """Entropía condicional de la distribución predictiva"""
    h = 0
    for p in predictive_distribution:
        if p > 0:
            h -= p * math.log2(p)
    return h

# Ejemplo de uso
scenarios_data = scenarios()
optimal_price, max_revenue = revenue_max([s['precio'] for s in scenarios_data])
print(f"Precio óptimo: {optimal_price}, Máxima recaudación: {max_revenue}")

# Simulación de entropía condicional ( ejemplo simplificado )
predictive_distribution = [0.5, 0.25, 0.125]
h = entropia_condicional(predictive_distribution)
print(f"Entropía condicional: {h}")