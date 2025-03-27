import cv2
import pytesseract
import numpy as np

# Configurar la ruta a l'executable de Tesseract si no està en el PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Descomentar i ajustar la ruta si cal

def augmentar_resolucio(imatge, escala=2.0):
    """Augmenta la resolució de la imatge utilitzant interpolació."""
    width = int(imatge.shape[1] * escala)
    height = int(imatge.shape[0] * escala)
    dim = (width, height)
    return cv2.resize(imatge, dim, interpolation=cv2.INTER_CUBIC)

def millorar_contrast(imatge):
    """Millora el contrast de la imatge per a una millor detecció."""
    # Equalització d'histograma per millorar el contrast
    imatge_eq = cv2.equalizeHist(imatge)
    
    # Aplica una lleugera reducció de soroll mantenint les vores
    imatge_filtrada = cv2.bilateralFilter(imatge_eq, 9, 75, 75)
    
    # Augmenta la nitidesa
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    imatge_nitida = cv2.filter2D(imatge_filtrada, -1, kernel)
    
    return imatge_nitida

def extraure_text_de_matricula(imatge_retallada):
    """Extrau el text de la matrícula utilitzant Tesseract OCR."""
    # Intentem diferents configuracions de preprocessament i Tesseract
    
    # Apliquem un llindar per separa millor el text del fons
    _, binaria1 = cv2.threshold(imatge_retallada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Prova alternativa amb threshold adaptatiu
    binaria2 = cv2.adaptiveThreshold(imatge_retallada, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Millorem la neteja d'imatge
    kernel = np.ones((2, 2), np.uint8)
    morfologia = cv2.morphologyEx(binaria1, cv2.MORPH_CLOSE, kernel)
    
    # Un altra tècnica: invertim la imatge
    binaria_inv = cv2.bitwise_not(binaria1)
    
    # Guardem les versions per depurar
    cv2.imwrite("preprocessat/binaria1.jpg", binaria1)
    cv2.imwrite("preprocessat/binaria2.jpg", binaria2)
    cv2.imwrite("preprocessat/morfologia.jpg", morfologia)
    cv2.imwrite("preprocessat/binaria_inv.jpg", binaria_inv)
    
    # Donem noms als mètodes de preprocessament per millor seguiment
    metodes_preprocessament = {
        "binaria_otsu": binaria1,
        "binaria_adaptativa": binaria2, 
        "morfologia": morfologia,
        "binaria_invertida": binaria_inv
    }
    
    # Intentem amb diferents configuracions de Tesseract
    configs = {
        "bloc_text": '--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        "linia_text": '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        "paraula": '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        "caracter": '--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    }
    
    # Provem cada configuració amb cada imatge processada
    resultats = []
    millor_resultat = {"text": "", "preprocessament": "", "config": "", "longitud": 0}
    
    print("\n--- TOTS ELS RESULTATS ---")
    for nom_prep, imatge in metodes_preprocessament.items():
        for nom_config, config in configs.items():
            try:
                text = pytesseract.image_to_string(imatge, config=config)
                text = text.strip().replace('\n', ' ')
                
                if text:  # Si s'ha detectat quelcom
                    print(f"[{nom_prep} + {nom_config}]: '{text}' (longitud: {len(text)})")
                    resultats.append({
                        "text": text,
                        "preprocessament": nom_prep,
                        "config": nom_config,
                        "longitud": len(text)
                    })
                    
                    # Actualitzem el millor resultat si aquest és més llarg
                    if len(text) > millor_resultat["longitud"]:
                        millor_resultat = {
                            "text": text,
                            "preprocessament": nom_prep,
                            "config": nom_config,
                            "longitud": len(text)
                        }
            except Exception as e:
                print(f"Error en Tesseract: {e}")
    
    print("\n--- MILLOR RESULTAT ---")
    if millor_resultat["longitud"] > 0:
        print(f"Text: '{millor_resultat['text']}'")
        print(f"Mètode de preprocessament: {millor_resultat['preprocessament']}")
        print(f"Configuració de Tesseract: {millor_resultat['config']}")
        print(f"Longitud: {millor_resultat['longitud']}")
        return millor_resultat["text"]
    else:
        print("No s'ha pogut detectar cap text")
        return "No s'ha pogut detectar text"

def main():
    imatge = cv2.imread("matricula2.jpg")
    if imatge is None:
        print("Imatge no trobada!")
        return
    
    imatge_gris = cv2.cvtColor(imatge, cv2.COLOR_BGR2GRAY)
    
    # Millora la imatge abans d'extraure el text (activem totes les millores)
    imatge_millorada = millorar_contrast(imatge_gris)
    imatge_gran = augmentar_resolucio(imatge_millorada, 3.0)
    
    # Guardar imatge processada per inspeccionar-la
    #cv2.imwrite("imatge_processada.jpg", imatge_gran)
    
    # Mostra la imatge processada
    #cv2.imshow("Imatge Original", imatge)
    #cv2.imshow("Imatge Processada", imatge_gran)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    text_matricula = extraure_text_de_matricula(imatge_gran)
    print("Matrícula detectada:", text_matricula)

if __name__ == "__main__":
    main()
