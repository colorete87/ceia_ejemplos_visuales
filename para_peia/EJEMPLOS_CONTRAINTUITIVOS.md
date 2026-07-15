# Ejemplos Contraintuitivos de Probabilidad

Los ejemplos contraintuitivos en probabilidad y estadística son escenarios donde la lógica matemática produce resultados que chocan fuertemente con el sentido común o la intuición cotidiana. Estos casos demuestran que nuestra intuición sobre el azar y el riesgo suele ser errónea, y nos recuerdan la importancia de utilizar herramientas matemáticas y análisis de datos rigurosos.

### 📌 Referencia de Preferencias
- 🟢 **Preferencia Alta**: Problemas recomendados prioritariamente como disparadores didácticos principales.
- 🟡 **Preferencia Media**: Problemas complementarios o de menor prioridad para el material didáctico.

A continuación, se presentan algunos de los ejemplos más famosos y sorprendentes:

---

## 1. La Paradoja del Cumpleaños 🟢 *(Preferencia: Alta)*
* ¿Cuántas personas se necesitan en una sala para que la probabilidad de que al menos dos cumplan años el mismo día supere el 50%?
* **El resultado contraintuitivo**: Solo se necesitan **23 personas**. Con 57 personas, la probabilidad supera el 99%.
* **Por qué ocurre**: Tendemos a pensar de manera egocéntrica (la probabilidad de que alguien cumpla años el mismo día que *yo*). Sin embargo, la paradoja busca *cualquier* pareja que comparta cumpleaños. En un grupo de 23 personas, hay $\frac{23 \times 22}{2} = 253$ parejas posibles que comparar, lo cual es mucho mayor de lo que el cerebro estima intuitivamente.

---

## 2. El Problema de Monty Hall 🟢 *(Preferencia: Alta)*
* Estás en un concurso y tienes tres puertas frente a ti. Detrás de una hay un automóvil y detrás de las otras dos hay cabras. Eliges la Puerta 1. El presentador (quien sabe qué hay detrás de cada puerta) abre la Puerta 3 revelando una cabra. Luego te ofrece la opción: "¿Quieres cambiar tu elección a la Puerta 2?". ¿Conviene cambiar?
* **El resultado contraintuitivo**: **Siempre conviene cambiar**. Al hacerlo, la probabilidad de ganar el automóvil sube de $\frac{1}{3}$ ($33.3\%$) a $\frac{2}{3}$ ($66.6\%$).
* **Por qué ocurre**: Tu elección inicial tiene una probabilidad de $\frac{1}{3}$ de ser correcta y una probabilidad de $\frac{2}{3}$ de estar en el grupo de las otras dos puertas. Cuando el presentador revela una cabra de entre las puertas no elegidas, el total de la probabilidad de ese grupo ($\frac{2}{3}$) se concentra por completo en la puerta restante que no elegiste originalmente.

---

## 3. La Falacia del Fiscal / Falsos Positivos (Pruebas Médicas) 🟢/🟡 *(Preferencia: Alta / Media)*
* Una prueba médica para detectar una enfermedad muy rara (afecta a 1 de cada 1.000 personas) tiene una precisión diagnóstica del 99% (tasa de verdaderos positivos y verdaderos negativos). Si te haces el test y da positivo, ¿cuál es la probabilidad de que realmente estés enfermo?
* **El resultado contraintuitivo**: La probabilidad real de estar enfermo es sorprendentemente baja, de aproximadamente **9%** (menos del 10%).
* **Por qué ocurre**: Imaginemos evaluar a 100.000 personas.
  - **Enfermos reales**: 100 personas (de las cuales 99 darán positivo).
  - **Sanos reales**: 99.900 personas. Dado que el test falla en el 1% de los casos, dará aproximadamente 999 falsos positivos.
  - El total de positivos es $99 + 999 = 1.098$. De todos los que dieron positivo, solo 99 están verdaderamente enfermos: $\frac{99}{1.098} \approx 9\%$.

---

## 4. La Paradoja de Simpson 🟡 *(Preferencia: Media)*
* Un tratamiento médico parece tener una tasa de éxito menor que otro al observar los datos agregados (hombres y mujeres combinados). Sin embargo, al segmentar los datos por género, el primer tratamiento resulta ser más efectivo tanto en hombres como en mujeres.
* **El resultado contraintuitivo**: La relación observada en múltiples grupos de datos puede invertirse por completo al combinar los grupos.
* **Por qué ocurre**: Se debe a la presencia de variables de confusión no controladas y a tamaños de muestra desiguales entre los grupos. Si un tratamiento se asigna de manera desproporcionada a casos más graves (o a un género con menor tasa de supervivencia base), su promedio agregado se verá penalizado artificialmente.

---

## 5. El Problema de los 100 Prisioneros 🟢 *(Preferencia: Alta)*
* Hay 100 prisioneros numerados del 1 al 100. En una habitación hay 100 cajas cerradas, cada una con un número del 1 al 100 en su interior, distribuidos al azar. Cada prisionero puede entrar a la habitación y abrir hasta 50 cajas para buscar su propio número. Deben dejar la habitación tal como la encontraron. Si todos encuentran su número, son liberados. Si uno solo falla, todos son ejecutados. No hay comunicación permitida una vez iniciado el juego.
* **El resultado contraintuitivo**: Si eligen cajas al azar, la probabilidad de éxito de que todos se salven es infinitesimal: $(\frac{1}{2})^{100} \approx 0.0000000000000000000000000000008\%$. Sin embargo, utilizando una estrategia basada en teoría de grafos, las probabilidades aumentan al **31.18%**.
* **Por qué ocurre (Estrategia de Ciclos)**: Cada prisionero comienza abriendo la caja marcada con su propio número por fuera, lee el papel interior, y va a abrir la caja con ese número, repitiendo el proceso hasta encontrar su número o alcanzar el límite de 50. Como los números forman lazos o ciclos cerrados, esta estrategia garantiza que el prisionero encontrará su número si la longitud máxima del ciclo en el que está es menor o igual a 50. La probabilidad de que la permutación aleatoria de las cajas no contenga ningún ciclo mayor a 50 es del 31.18%.

---

## 6. La Paradoja de los Dos Sobres 🟢 *(Preferencia: Alta)*
* Se te presentan dos sobres cerrados que contienen dinero. Uno de ellos tiene exactamente el doble de dinero que el otro. Eliges uno, lo abres y contiene \$100. Antes de guardarlo, se te da la opción de cambiarlo por el otro. ¿Tiene sentido cambiar?
  - El cálculo intuitivo de valor esperado sugiere que en el otro sobre puede haber \$50 (con probabilidad 0.5) o \$200 (con probabilidad 0.5).
  - El valor esperado al cambiar sería: $0.5 \times \$50 + 0.5 \times \$200 = \$125$. Como \$125 es mayor que \$100, parece que siempre conviene cambiar. Pero esto generaría un bucle infinito si se aplicara antes de abrir el sobre.
* **El resultado contraintuitivo**: Estadísticamente, **no hay ninguna ventaja** en cambiar; el valor esperado real es el mismo.
* **Por qué ocurre**: La falacia radica en tratar los dos posibles estados del mundo como si pertenecieran al mismo espacio de probabilidad con la misma base. Si la cantidad menor es $x$, los sobres contienen $x$ y $2x$ (el total fijo es $3x$). Al abrir y ver \$100, los dos escenarios posibles son independientes:
  - Escenario A: Los sobres son \$50 y \$100 (el total es \$150, elegiste el mayor).
  - Escenario B: Los sobres son \$100 y \$200 (el total es \$300, elegiste el menor).
  Las dos situaciones pertenecen a distribuciones de probabilidad a priori diferentes sobre el dinero total que hay en la mesa, por lo que el cálculo de valor esperado simple asumiendo probabilidad simétrica de 0.5 sobre valores condicionales cambiantes es incorrecto.
