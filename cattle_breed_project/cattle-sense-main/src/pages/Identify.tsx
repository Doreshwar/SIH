import { useState, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Camera, Upload, Loader2, Download, Share2, AlertTriangle } from 'lucide-react';
import { API_BASE_URL } from '@/lib/mockData';
import { useToast } from '@/hooks/use-toast';

interface PredictionResult {
  breed_name: string;
  confidence: number;
  breed_id: string;
  characteristics: {
    origin: string;
    primary_use: string;
    average_weight: string;
    distinctive_features: string;
  };
}

interface APIResponse {
  success: boolean;
  predictions: PredictionResult[];
  message?: string;
  model_loaded?: boolean;
}

const Identify = () => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState<PredictionResult[]>([]);
  const [apiMessage, setApiMessage] = useState<string>('');
  const [modelLoaded, setModelLoaded] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (file.size > 10 * 1024 * 1024) {
        toast({
          title: "File too large",
          description: "Please select an image under 10MB",
          variant: "destructive"
        });
        return;
      }
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResults([]);
    }
  };

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      if (file.size > 10 * 1024 * 1024) {
        toast({
          title: "File too large",
          description: "Please select an image under 10MB",
          variant: "destructive"
        });
        return;
      }
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResults([]);
    }
  };

  const processImage = async () => {
    if (!selectedImage) return;
    
    setIsProcessing(true);
    
    try {
      // Create FormData for file upload
      const formData = new FormData();
      formData.append('image', selectedImage);
      
      // Call the ML backend API
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data: APIResponse = await response.json();
      
      if (data.success && data.predictions) {
        setResults(data.predictions);
        setApiMessage(data.message || '');
        setModelLoaded(data.model_loaded || false);
        
        const statusIcon = data.model_loaded ? '✅' : '🤖';
        const statusText = data.model_loaded ? 'Real AI Model' : 'Demo Mode';
        
        toast({
          title: `${statusIcon} Identification Complete`,
          description: `${statusText}: Found ${data.predictions.length} possible matches`
        });
      } else {
        throw new Error(data.message || 'Unknown error occurred');
      }
      
    } catch (error) {
      console.error('Prediction error:', error);
      toast({
        title: "Identification Failed",
        description: error instanceof Error ? error.message : "Failed to identify cattle breed. Please try again.",
        variant: "destructive"
      });
      
      // Show offline fallback message
      setResults([{
        breed_name: "Connection Error",
        confidence: 0,
        breed_id: "error",
        characteristics: {
          origin: "N/A",
          primary_use: "Please ensure the backend server is running",
          average_weight: "N/A",
          distinctive_features: "Check if the Flask API is accessible at http://localhost:5000"
        }
      }]);
    } finally {
      setIsProcessing(false);
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return "text-green-600";
    if (confidence >= 0.6) return "text-yellow-600";
    return "text-red-500";
  };

  return (
    <div className="min-h-screen gradient-subtle">
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold text-foreground mb-4">
              Identify Your Cattle
            </h1>
            <p className="text-lg text-muted-foreground mb-4">
              Upload or capture an image to identify the breed with AI precision
            </p>
            {apiMessage && (
              <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm ${
                modelLoaded 
                  ? 'bg-green-100 text-green-800 border border-green-200' 
                  : 'bg-blue-100 text-blue-800 border border-blue-200'
              }`}>
                <span className="w-2 h-2 rounded-full bg-current"></span>
                {modelLoaded ? '✅ Real AI Model Active' : '🤖 Demo Mode Active'}
              </div>
            )}
          </div>

          {/* Upload Interface */}
          <Card className="mb-8 shadow-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Upload className="w-5 h-5" />
                Upload Cattle Image
              </CardTitle>
            </CardHeader>
            <CardContent>
              {!selectedImage ? (
                <div
                  className="border-2 border-dashed border-border rounded-xl p-12 text-center hover:border-primary transition-smooth cursor-pointer"
                  onDrop={handleDrop}
                  onDragOver={(e) => e.preventDefault()}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Camera className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-lg font-medium text-foreground mb-2">
                    Drag and drop your image here
                  </p>
                  <p className="text-muted-foreground mb-4">
                    or click to browse files
                  </p>
                  <p className="text-sm text-muted-foreground">
                    Supports JPG, PNG, WebP (Max 10MB)
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="relative">
                    <img
                      src={previewUrl}
                      alt="Selected cattle"
                      className="w-full max-w-md mx-auto rounded-lg shadow-card"
                    />
                  </div>
                  <div className="flex justify-center gap-3">
                    <Button 
                      variant="outline" 
                      onClick={() => {
                        setSelectedImage(null);
                        setPreviewUrl('');
                        setResults([]);
                      }}
                    >
                      Remove Image
                    </Button>
                    <Button 
                      variant="farm"
                      onClick={processImage}
                      disabled={isProcessing}
                    >
                      {isProcessing ? (
                        <>
                          <Loader2 className="w-4 h-4 animate-spin" />
                          Processing...
                        </>
                      ) : (
                        'Identify Breed'
                      )}
                    </Button>
                  </div>
                </div>
              )}
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                className="hidden"
              />
            </CardContent>
          </Card>

              {results.length > 0 && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <h2 className="text-2xl font-bold text-foreground">
                  Identification Results
                </h2>
                <div className="flex gap-2">
                  <Button variant="outline" size="sm">
                    <Download className="w-4 h-4" />
                    Export
                  </Button>
                  <Button variant="outline" size="sm">
                    <Share2 className="w-4 h-4" />
                    Share
                  </Button>
                </div>
              </div>
              
              <div className="grid gap-4">
                {results.map((result, index) => (
                  <Card key={`${result.breed_id}-${index}`} className="shadow-card hover:shadow-elegant transition-smooth">
                    <CardContent className="p-6">
                      {result.breed_id === 'error' ? (
                        // Error state display
                        <div className="flex items-start gap-4">
                          <AlertTriangle className="w-8 h-8 text-red-500 mt-1" />
                          <div className="flex-1">
                            <h3 className="text-xl font-semibold text-red-600 mb-2">
                              {result.breed_name}
                            </h3>
                            <div className="space-y-2 text-sm">
                              <p className="text-muted-foreground">{result.characteristics.primary_use}</p>
                              <p className="text-muted-foreground">{result.characteristics.distinctive_features}</p>
                            </div>
                          </div>
                        </div>
                      ) : (
                        // Normal prediction display
                        <>
                          <div className="flex items-start justify-between mb-4">
                            <div>
                              <h3 className="text-xl font-semibold text-foreground">
                                {result.breed_name}
                              </h3>
                              <Badge variant="secondary" className="mt-1">
                                {index === 0 ? 'Best Match' : `Alternative ${index + 1}`}
                              </Badge>
                            </div>
                            <div className="text-right">
                              <div className={`text-2xl font-bold ${getConfidenceColor(result.confidence)}`}>
                                {Math.round(result.confidence * 100)}%
                              </div>
                              <p className="text-sm text-muted-foreground">Confidence</p>
                            </div>
                          </div>
                          
                          <Progress 
                            value={result.confidence * 100} 
                            className="mb-4"
                          />
                          
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                            <div>
                              <p className="font-medium text-foreground">Origin</p>
                              <p className="text-muted-foreground">{result.characteristics.origin}</p>
                            </div>
                            <div>
                              <p className="font-medium text-foreground">Primary Use</p>
                              <p className="text-muted-foreground">{result.characteristics.primary_use}</p>
                            </div>
                            <div>
                              <p className="font-medium text-foreground">Weight</p>
                              <p className="text-muted-foreground">{result.characteristics.average_weight}</p>
                            </div>
                            <div>
                              <p className="font-medium text-foreground">Features</p>
                              <p className="text-muted-foreground">{result.characteristics.distinctive_features}</p>
                            </div>
                          </div>
                        </>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Identify;