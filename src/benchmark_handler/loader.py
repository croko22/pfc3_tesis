import os

def load_sf110_benchmark(benchmark_path: str):
    """
    Carga el benchmark SF110 desde el disco.

    Itera sobre todos los proyectos en el benchmark y produce (yields)
    la ruta y el contenido de cada archivo .java de código fuente.

    Args:
        benchmark_path: La ruta raíz al directorio SF110 descomprimido.

    Yields:
        Un iterador de tuplas (file_path, file_content) para cada archivo
        de código fuente (NO de prueba) encontrado.
    """
    if not os.path.isdir(benchmark_path):
        print(f"Error: La ruta del benchmark no existe: {benchmark_path}")
        return

    print(f"Iniciando carga de benchmark desde: {benchmark_path}")

    # SF110 está estructurado como una colección de proyectos,
    # cada uno en su propio directorio.
    for project_name in os.listdir(benchmark_path):
        project_path = os.path.join(benchmark_path, project_name)

        if os.path.isdir(project_path):
            # Usamos os.walk para recorrer recursivamente el árbol de directorios
            for root, dirs, files in os.walk(project_path):
                
                # 1. Filtrar directorios de prueba
                # No queremos procesar las pruebas existentes, solo el código fuente.
                if 'src/test/java' in root or 'src/test' in root:
                    continue

                # 2. Procesar solo los archivos de código fuente
                if 'src/main/java' in root or 'src' in root:
                    for file in files:
                        if file.endswith(".java"):
                            
                            # 3. Filtrar archivos de prueba por nombre (doble seguridad)
                            if file.endswith("Test.java") or file.endswith("Tests.java"):
                                continue

                            file_path = os.path.join(root, file)
                            
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    file_content = f.read()
                                # Producimos la ruta y el contenido para ser procesados
                                yield (file_path, file_content)
                            except Exception as e:
                                print(f"Error al leer {file_path}: {e}")