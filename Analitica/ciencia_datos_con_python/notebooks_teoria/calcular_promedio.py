def calcular_promedio(lista: list[float]) -> float:
    """
    Calcula el promedio de una lista de números.

    Args:
        lista (list of int o float): Lista de valores numéricos.

    Returns:
        float: El promedio de los elementos de la lista.
               Si la lista está vacía, retorna None.
    """
    if len(lista) == 0:
        print("La lista está vacía, no se puede calcular el promedio.")
        return None  # Alternativamente, se podría lanzar una excepción.

    suma = 0
    for num in lista:
        suma += num
    promedio = suma / len(lista)
    return promedio

# Ejemplo de uso:
numeros = [10, 20, 30, 40, 50]

resultado = calcular_promedio(numeros)
print("El promedio es:", resultado)
