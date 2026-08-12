## Análisis de Complejidad Computacional

### 1. Endpoint `forecast`

**Complejidad temporal y espacial**: O(n log n) y por qué  
El endpoint `forecast` realiza una optimización de la función de costo para calibrar los parámetros p/q del modelo Bass Diffusion utilizando el MLE (Maximum Likelihood Estimation). La optimización se realiza mediante un algoritmo de búsqueda numérica, como BFGS o Newton-Raphson, que tiene una complejidad temporal de O(n log n), donde n es la longitud de la serie de tiempo. El espacio necesario es proporcional a la dimensión del problema, lo que también es O(n).

**Caso mejor / promedio / peor**:  
- Mejor: O(1) para casos en los que el conjunto de datos sea muy pequeño o el modelo ya esté bien calibrado.
- Promedio: O(log n)
- Peor: O(n)

**Cuello de botella identificado**: La optimización del MLE es el cuello de botella principal. Aunque la complejidad temporal es logarítmica, el tamaño del problema puede crecer rápidamente con el aumento de los datos.

### 2. Endpoint `forecast_with_entropy`

**Complejidad temporal y espacial**: O(n^2) y por qué  
El endpoint `forecast_with_entropy` realiza dos pasos principales: primero, genera una predicción numérica usando el modelo Bass Diffusion calibrado; segundo, calcula la entropía condicional H(X_{t+k} | X_{1..t}) para cada paso de horizonte. La generación del forecast tiene una complejidad lineal O(n). Calcular la entropía condicional para un paso k implica calcular la probabilidad distribuida condicionada, lo que puede ser cuadrático en el peor caso (O(k^2)). Sin embargo, como el cálculo de la entropía es necesario para cada horizonte, la complejidad total se multiplica por n.

**Caso mejor / promedio / peor**:  
- Mejor: O(n) para casos en los que el conjunto de datos sea muy pequeño y la serie temporal siga una distribución simple.
- Promedio: O(n^2)
- Peor: O(n^3)

**Cuello de botella identificado**: El cálculo de la entropía condicional es el cuello de botella principal. La complejidad cuadrática puede volverse inaceptable con conjuntos de datos grandes.

### Punto de Saturación Estimado

El punto de saturación estimado para esta primitiva es alrededor de 100 requests/segundo en condiciones ideales (con una máquina potente y datos pequeños). Con datos más grandes o con mayor complejidad, el punto de saturación disminuirá.

### Estrategia de Optimización para Escalar Más Allá

- **Paralelización**: Utilizar procesadores adicionales o incluso GPU para paralelizar partes del algoritmo que pueden hacerlo eficientemente.
- **Sampling Approximation**: En lugar de calcular la entropía condicional exacta, usar métodos de muestreo o aproximación para estimarla. Esto puede reducir significativamente el tiempo de cálculo sin sacrificar demasiado precisión.
- **Caching**: Cachear resultados recientes y evitar recalculaciones innecesarias.
- **Asincronismo**: Implementar endpoints asíncronos para procesos que pueden llevar mucho tiempo, permitiendo que la API maneje múltiples peticiones simultáneamente.

Este análisis de complejidad ayudará a entender cómo el endpoint `forecast_with_entropy` puede escalar con diferentes conjuntos de datos y configuraciones, lo que es crucial para su implementación en un entorno de producción.