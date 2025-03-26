# Informe de Procesamiento de Imágenes

## gato

### Conversión a Escala de Grises
La imagen fue convertida con éxito a escala de grises.

### Rango de Valores de Píxeles
- Valor mínimo de píxel: 0
- Valor máximo de píxel: 221

### Normalización
La imagen fue normalizada para mejorar el contraste utilizando la fórmula:
`normalizada = (imagen - min) / (max - min) * 255`

### Resultados de Filtrado
- Filtro de Media (tamaño de kernel = 3): Redujo el ruido mientras preservaba los bordes, pero causó cierto desenfoque.
- Filtro de Mediana (tamaño de kernel = 3): Eliminó eficazmente el ruido de sal y pimienta mientras preservaba mejor los bordes que el filtrado de media.
- Filtro de Máximo (tamaño de kernel = 3): Realzó características brillantes y expandió regiones claras.
- Filtro de Mínimo (tamaño de kernel = 3): Realzó características oscuras y expandió regiones oscuras.

---

## radiografía_de_tórax

### Conversión a Escala de Grises
La imagen fue convertida con éxito a escala de grises.

### Rango de Valores de Píxeles
- Valor mínimo de píxel: 0
- Valor máximo de píxel: 255

### Normalización
La imagen fue normalizada para mejorar el contraste utilizando la fórmula:
`normalizada = (imagen - min) / (max - min) * 255`

### Resultados de Filtrado
- Filtro de Media (tamaño de kernel = 3): Redujo el ruido mientras preservaba los bordes, pero causó cierto desenfoque.
- Filtro de Mediana (tamaño de kernel = 3): Eliminó eficazmente el ruido de sal y pimienta mientras preservaba mejor los bordes que el filtrado de media.
- Filtro de Máximo (tamaño de kernel = 3): Realzó características brillantes y expandió regiones claras.
- Filtro de Mínimo (tamaño de kernel = 3): Realzó características oscuras y expandió regiones oscuras.

---

## bosque

### Conversión a Escala de Grises
La imagen fue convertida con éxito a escala de grises.

### Rango de Valores de Píxeles
- Valor mínimo de píxel: 0
- Valor máximo de píxel: 251

### Normalización
La imagen fue normalizada para mejorar el contraste utilizando la fórmula:
`normalizada = (imagen - min) / (max - min) * 255`

### Resultados de Filtrado
- Filtro de Media (tamaño de kernel = 3): Redujo el ruido mientras preservaba los bordes, pero causó cierto desenfoque.
- Filtro de Mediana (tamaño de kernel = 3): Eliminó eficazmente el ruido de sal y pimienta mientras preservaba mejor los bordes que el filtrado de media.
- Filtro de Máximo (tamaño de kernel = 3): Realzó características brillantes y expandió regiones claras.
- Filtro de Mínimo (tamaño de kernel = 3): Realzó características oscuras y expandió regiones oscuras.

---

## papiro

### Conversión a Escala de Grises
La imagen fue convertida con éxito a escala de grises.

### Rango de Valores de Píxeles
- Valor mínimo de píxel: 1
- Valor máximo de píxel: 255

### Normalización
La imagen fue normalizada para mejorar el contraste utilizando la fórmula:
`normalizada = (imagen - min) / (max - min) * 255`

### Resultados de Filtrado
- Filtro de Media (tamaño de kernel = 3): Redujo el ruido mientras preservaba los bordes, pero causó cierto desenfoque.
- Filtro de Mediana (tamaño de kernel = 3): Eliminó eficazmente el ruido de sal y pimienta mientras preservaba mejor los bordes que el filtrado de media.
- Filtro de Máximo (tamaño de kernel = 3): Realzó características brillantes y expandió regiones claras.
- Filtro de Mínimo (tamaño de kernel = 3): Realzó características oscuras y expandió regiones oscuras.

---

## Conclusiones

### Normalización
La normalización mejoró la calidad de la imagen utilizando el rango dinámico completo de valores de píxeles (0-255). Esta mejora es particularmente notable en imágenes con bajo contraste, donde los valores de píxeles están concentrados en un rango estrecho. Al estirar este rango para cubrir 0-255, podemos hacer que los detalles sean más visibles para el ojo humano.

### Efectos de Filtrado
- **Filtro de Media**: Proporciona buena reducción de ruido pero tiende a desenfocar bordes y detalles finos. Es adecuado para imágenes con distribución de ruido tipo gaussiano.
- **Filtro de Mediana**: Excelente para eliminar ruido de sal y pimienta mientras preserva mejor los bordes que el filtrado de media. Es menos efectivo contra el ruido gaussiano.
- **Filtro de Máximo**: Útil para encontrar los puntos más brillantes en una imagen y realzar características brillantes. Puede ser utilizado para detectar objetos claros sobre fondos oscuros.
- **Filtro de Mínimo**: Útil para encontrar los puntos más oscuros en una imagen y realzar características oscuras. Puede ser utilizado para detectar objetos oscuros sobre fondos claros.

### Justificación de Parámetros
Se eligió un tamaño de kernel de 3x3 para todos los filtros ya que proporciona un buen equilibrio entre reducción de ruido y preservación de detalles. Tamaños de kernel más grandes resultarían en un filtrado más agresivo pero a costa de perder detalles importantes de la imagen. Para aplicaciones que requieran una reducción de ruido más fuerte, se podrían considerar tamaños de kernel más grandes.