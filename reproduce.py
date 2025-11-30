import subprocess
import sys
import os

def run_script(description, script_path):
    print(f"Ejecutando: {description}")
    try:
        if not os.path.exists(script_path):
            print(f"Error: Archivo no encontrado - {script_path}")
            return False
            
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Completado: {description}")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"Error en: {description}")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"Excepcion en {description}: {e}")
        return False

def main():
    print("Iniciando pipeline completo del sistema predictivo")
    print("=" * 60)
    
    scripts = [
        ("Generacion de datos sinteticos", "src/data_generation.py"),
        ("Calculo de caracteristicas", "src/features.py"),
        ("Entrenamiento de modelos", "src/models.py")
    ]
    
    success_count = 0
    for description, script_path in scripts:
        if run_script(description, script_path):
            success_count += 1
        print("-" * 40)
    
    print("=" * 60)
    if success_count == len(scripts):
        print("Pipeline completado exitosamente")
        print("\nResumen:")
        print("- Datos sinteticos generados")
        print("- Caracteristicas calculadas")
        print("- Modelos entrenados")
        print("\nPara iniciar el dashboard ejecuta:")
        print("streamlit run app/streamlit_app.py")
    else:
        print(f"Completados {success_count} de {len(scripts)} pasos")
        print("Revise los errores anteriores")

if __name__ == "__main__":
    main()