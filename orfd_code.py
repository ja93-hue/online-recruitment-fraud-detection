import zipfile
import os

def zip_code_only(zip_name="orfd_code.zip"):
    exclude_dirs = {"data", "logs", "models", "__pycache__"}
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk("app"):
            # Exclude unwanted directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for file in files:
                if file.endswith(".py") or file.endswith(".md") or file.endswith(".txt") or file.endswith(".ini"):
                    filepath = os.path.join(root, file)
                    zipf.write(filepath)
        # Optionally add requirements.txt and README.md from root
        for fname in ["requirements.txt", "README.md"]:
            if os.path.exists(fname):
                zipf.write(fname)
    print(f"Created {zip_name}")

zip_code_only()