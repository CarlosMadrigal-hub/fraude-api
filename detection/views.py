from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
import os
from .ml_utils import perform_kmeans_analysis

class KMeansFraudDetectionView(APIView):
    def post(self, request):
        n_clusters = request.data.get('n_clusters')

        if not n_clusters:
            return Response({"error": "Falta 'n_clusters'."}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            n_clusters = int(n_clusters)
            if not (2 <= n_clusters <= 20):
                return Response({"error": "Rango válido: 2-20."}, status=status.HTTP_400_BAD_REQUEST)
        except ValueError:
            return Response({"error": "Debe ser un número entero."}, status=status.HTTP_400_BAD_REQUEST)

        # CAMBIO AQUÍ: Apuntamos al archivo ZIP
        zip_path = os.path.join(settings.BASE_DIR, 'detection', 'creditcard.zip')
        
        try:
            results = perform_kmeans_analysis(n_clusters, zip_path)
            
            if "error" in results:
                # Si devuelve error (ej. no encontró archivo), enviamos 500 o 404
                return Response(results, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
            return Response(results, status=status.HTTP_200_OK)

        except Exception as e:
            return Response(
                {"error": f"Error interno: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )