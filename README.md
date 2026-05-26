# Signal Lab: Predicción de Riesgo de Diabetes Tipo 2 en Adultos Mexicanos

Este repositorio contiene la arquitectura de software, el pipeline de ingeniería de datos y los modelos de Inteligencia Artificial desarrollados para el tamizaje preventivo y diagnóstico temprano de Diabetes Tipo 2, utilizando como fuente analítica los microdatos de la **ENSANUT (Encuesta Nacional de Salud y Nutrición) Continua 2024**.

Proyecto diseñado, depurado e implementado bajo los estándares de evaluación y excelencia técnica global de **Samsung Innovation Campus** en colaboración con la **Universidad de Monterrey (UDEM)**.

---

## Despliegue e Interfaz Visual
Para explorar la metodología analítica detallada, la visualización interactiva de los resultados y el comportamiento de las métricas en producción, visita nuestro sitio oficial:
**[Sitio de Producción del Proyecto](https://byf1nd.github.io/deteccion-diabetes-ia/)**

---

## Stack Tecnológico e Infraestructura
**Lenguaje Core:** Python 3.10+ (Optimización matricial y desarrollo analítico).
**Procesamiento de Big Data:** Pandas, NumPy (Ingesta y ETL de datasets nacionales masivos).
**Modelado Predictivo y Segmentación:** Scikit-Learn (Algoritmos supervisados y clustering K-Means)
**Entornos de Ejecución:** Google Colab / Jupyter Notebooks.
**Análisis Gráfico:** Matplotlib, Seaborn (Análisis de correlación e importancia de características).

---

## Arquitectura de Directorios (Estructura del Proyecto)

El repositorio sigue un diseño modular y limpio, separando las fuentes de datos crudos de los flujos lógicos de código y los entregables analíticos:

```text
├── data/           # Microdatos crudos e históricos de la ENSANUT 2024 (Estructuras .csv y .zip)
├── notebooks/      # Código fuente documentado en cuadernos interactivos de Python (.ipynb)
├── reports/        # Reportes analíticos finales (PDFs) y artefactos gráficos del modelo (.png)
├── .gitignore      # Exclusión de archivos basura y dependencias del sistema operativo
├── _config.yml     # Configuración del entorno de despliegue para GitHub Pages
├── index.md        # Punto de entrada de la interfaz web del proyecto
└── README.md       # Documentación técnica principal del sistema

Componentes Clave del Pipeline Analítico
	1.	Ingesta y Data Profiling: Extracción de datos masivos nacionales desde las estructuras relacionales complejas de la encuesta gubernamental.
	2.	Pipeline ETL y Limpieza Estructural: Normalización de tipos de datos, imputación avanzada de valores nulos o atípicos, y estandarización de variables antropométricas y de tensión arterial.
	3.	Resolución de Incidentes Críticos (Debugging): Scripts de rescate automatizado para corregir excepciones lógicas de desalineación de folios e inconsistencias de registros (KeyErrors).
	4.	Segmentación Predictiva: Modelado analítico multifactorial enfocado en maximizar la sensibilidad y el F1-Score para mitigar falsos negativos en el tamizaje de salud pública.
Responsabilidades y Contribuciones Técnicas Destacadas
Alberto Plascencia — Ingeniero de Datos y Optimización Algorítmica
Como responsable del núcleo de infraestructura de datos y rendimiento de código dentro del equipo de Signal Lab, diseñé y ejecuté los siguientes componentes analíticos:
•	Ingeniería de Datos a Gran Escala: Lógica de extracción, transformación y carga (ETL) para bases de datos masivas nacionales de la ENSANUT, gestionando matrices y estructuras dinámicas complejas.
•	Scripting de Rescate y Debugging Crítico: Diagnóstico y corrección automatizada de fallas de integración y desalineación de folios analíticos (KeyErrors). Desarrollé scripts correctivos optimizados en Python que rescataron con éxito más de 24,000 registros de datos corruptos, habilitando un set de entrenamiento íntegro para el modelo.
•	Optimización de Complejidad Temporal: Reducción estricta de la latencia de procesamiento en los bucles de manipulación matricial de los datos aplicando principios de Big O Notation, garantizando flujos eficientes en la preparación del dataset.
Equipo de Ingeniería (Signal Lab)
•	Medina Mixtega Ángel Miguél — Desarrollo de Modelos e Integración
•	Plascencia Arevalo Alberto — Ingeniería de Datos y Optimización
•	Méndez Ortega Paula — Análisis Estadístico y Documentación
•	Mota Barraza Moisés — Visualización y Despliegue Web
•	Mendez Damián Brandon Efren — Validación y Control de Calidad
Este proyecto de software representa la culminación de los módulos avanzados de Inteligencia Artificial desarrollados bajo la mentoría global de Samsung Innovation Campus.