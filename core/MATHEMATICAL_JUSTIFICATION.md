### 1. Por qué máximo 5 endpoints (Hick's Law: $T = b \cdot \log_2(n+1)$)
Según Hick's Law, la complejidad de una interfaz de usuario aumenta con el número de opciones disponibles. Cada endpoint representa una decisión que puede requerir tiempo y esfuerzo para aprender y usar. Limitando a 5 endpoints, minimizamos la curva de aprendizaje y facilitamos la adopción del API por parte del desarrollador.

### 2. Por qué pricing per-call vs por asiento (elasticidad precio-demanda)
El modelo de precios por llamada permite una mayor flexibilidad y transparencia en el costo. Los usuarios solo pagan por las operaciones realizadas, lo que favorece la escalabilidad y la eficiencia económica.

### 3. Por qué esta estructura de datos específica (complejidad algorítmica)
La estructura de datos utilizada para almacenar los parámetros del modelo Bass incluye una matriz $P$ para las probabilidades condicionales y un vector $\theta = [p, q, M]$ para los parámetros de la distribución. Esta estructura permite una representación compacta y eficiente de los cálculos necesarios.

### 4. El invariante matemático que hace esta solución correcta
El invariante matemático es $H(X_{t+k} | X_{1..t}) = H(p|X_{1..t},q,M) + H(q|p,X_{1..t},M) + H(M|p,q,X_{1..t})$, donde $H$ representa la entropía condicional. Este invariante asegura que la entropía acumulada es consistente y refleja la confiabilidad del forecast.

### 5. Límites teóricos del sistema (qué no puede hacer y por qué)
El modelo Bass está limitado en su capacidad de manejar series temporales muy cortas debido a sus parámetros $p$ y $q$. Para series con menos de 15 puntos, el valor de $M$ debe ser calibrado manualmente. Además, los modelos basados en difusión no capturan patrones ocultos o estacionariedad que podrían estar presentes en la serie temporal.

### Justificación Matemática
La combinación de Bass Diffusion con Entropía de Shannon permite una predicción dual de la adopción y la confiabilidad del forecast. El modelo Bass calibrado vía MLE proporciona una estimación precisa de los parámetros $p$ y $q$, que son cruciales para el cálculo de la curva de adopción. La entropía condicional, en cambio, mide la incertidumbre residual del modelo sobre las futuras observaciones, lo que permite al cliente determinar cuándo el forecast es confiable y cuándo comienza a extrapolarse.

La entropía $H(X_{t+k}|X_{1..t})$ se calcula como:
$$ H(X_{t+k}|X_{1..t}) = -\sum_{i=1}^{n} P(X_{t+k}=x_i|X_{1..t}) \log_2 P(X_{t+k}=x_i|X_{1..t}) $$

Donde $P(X_{t+k}=x_i|X_{1..t})$ es la probabilidad condicional de que el evento $x_i$ ocurra en el futuro, dada la historia pasada $X_{1..t}$.

El invariante matemático asegura la consistencia de los cálculos:
$$ H(X_{t+k} | X_{1..t}) = H(p|X_{1..t},q,M) + H(q|p,X_{1..t},M) + H(M|p,q,X_{1..t}) $$

Este enfoque no solo proporciona una predicción numérica, sino también una métrica de confiabilidad que permite al cliente trunca el horizonte donde la información marginal se ve reducida a un nivel aceptable.