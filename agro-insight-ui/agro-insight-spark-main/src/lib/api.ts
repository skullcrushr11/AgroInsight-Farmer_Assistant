const API_BASE_URL = 'http://127.0.0.1:8000/api';

export const api = {
  async chat(message: string) {
    try {
      console.log("API chat method called with message:", message);
      
      // Validate message
      if (!message || typeof message !== 'string') {
        throw new Error("Invalid message format");
      }

      const trimmedMessage = message.trim();
      if (!trimmedMessage) {
        throw new Error("Empty message after trimming");
      }

      const requestBody = { 
        message: trimmedMessage,
        role: "user"
      };
      
      console.log("Preparing request to:", `${API_BASE_URL}/chat`);
      console.log("Request body:", JSON.stringify(requestBody, null, 2));
      
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });
      
      console.log("Response status:", response.status);
      const responseText = await response.text();
      console.log("Raw response:", responseText);
      
      if (!response.ok) {
        let errorMessage = 'Failed to send message';
        try {
          const errorJson = JSON.parse(responseText);
          if (errorJson.detail) {
            errorMessage = typeof errorJson.detail === 'string' 
              ? errorJson.detail 
              : JSON.stringify(errorJson.detail);
          } else if (errorJson.error) {
            errorMessage = errorJson.error;
          }
        } catch (e) {
          errorMessage = responseText || errorMessage;
        }
        throw new Error(errorMessage);
      }
      
      try {
        const parsedResponse = JSON.parse(responseText);
        console.log("Parsed response:", parsedResponse);
        return parsedResponse;
      } catch (e) {
        console.error("Failed to parse response:", e);
        throw new Error("Invalid response from server");
      }
    } catch (error) {
      console.error("Chat API Error:", error);
      if (error instanceof Error) {
        throw error;
      }
      throw new Error("An unknown error occurred");
    }
  },

  async cropRecommendation(data: {
    N: number;
    P: number;
    K: number;
    temperature: number;
    humidity: number;
    ph: number;
    rainfall: number;
  }) {
    try {
      console.log("Sending crop recommendation request to:", `${API_BASE_URL}/crop-recommendation`);
      console.log("Request data:", JSON.stringify(data, null, 2));
      
      const response = await fetch(`${API_BASE_URL}/crop-recommendation`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify(data),
      });
      
      console.log("Response status:", response.status);
      const responseText = await response.text();
      console.log("Raw response:", responseText);
      
      if (!response.ok) {
        let errorMessage = 'Failed to get crop recommendation';
        try {
          const errorJson = JSON.parse(responseText);
          if (errorJson.detail) {
            errorMessage = typeof errorJson.detail === 'string' 
              ? errorJson.detail 
              : JSON.stringify(errorJson.detail);
          } else if (errorJson.error) {
            errorMessage = errorJson.error;
          }
        } catch (e) {
          errorMessage = responseText || errorMessage;
        }
        throw new Error(errorMessage);
      }
      
      try {
        return JSON.parse(responseText);
      } catch (e) {
        console.error("Failed to parse response:", e);
        throw new Error("Invalid response from server");
      }
    } catch (error) {
      console.error("Crop Recommendation API Error:", error);
      if (error instanceof Error) {
        throw error;
      }
      throw new Error("An unknown error occurred");
    }
  },

  async fertilizerClassification(data: {
    soil_color?: string;
    nitrogen?: number;
    phosphorus?: number;
    potassium?: number;
    ph?: number;
    temperature?: number;
    crop?: string;
    moisture?: number;
    rainfall?: number;
    carbon?: number;
    soil_type?: string;
  }) {
    try {
      console.log("Sending fertilizer classification request to:", `${API_BASE_URL}/fertilizer-classification`);
      console.log("Request data:", JSON.stringify(data, null, 2));
      
      const response = await fetch(`${API_BASE_URL}/fertilizer-classification`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify(data),
      });
      
      console.log("Response status:", response.status);
      const responseText = await response.text();
      console.log("Raw response:", responseText);
      
      if (!response.ok) {
        let errorMessage = 'Failed to get fertilizer classification';
        try {
          const errorJson = JSON.parse(responseText);
          if (errorJson.detail) {
            errorMessage = typeof errorJson.detail === 'string' 
              ? errorJson.detail 
              : JSON.stringify(errorJson.detail);
          } else if (errorJson.error) {
            errorMessage = errorJson.error;
          }
        } catch (e) {
          errorMessage = responseText || errorMessage;
        }
        throw new Error(errorMessage);
      }
      
      try {
        return JSON.parse(responseText);
      } catch (e) {
        console.error("Failed to parse response:", e);
        throw new Error("Invalid response from server");
      }
    } catch (error) {
      console.error("Fertilizer Classification API Error:", error);
      if (error instanceof Error) {
        throw error;
      }
      throw new Error("An unknown error occurred");
    }
  },

  async yieldPrediction(data: {
    soil_type: string;
    crop: string;
    rainfall: number;
    temperature: number;
    fertilizer_used: number;
    irrigation_used: number;
    weather_condition: string;
    days_to_harvest: number;
  }) {
    try {
      console.log("Sending yield prediction request to:", `${API_BASE_URL}/yield-prediction`);
      console.log("Request data:", JSON.stringify(data, null, 2));
      
      const response = await fetch(`${API_BASE_URL}/yield-prediction`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify(data),
      });
      
      console.log("Response status:", response.status);
      const responseText = await response.text();
      console.log("Raw response:", responseText);
      
      if (!response.ok) {
        let errorMessage = 'Failed to get yield prediction';
        try {
          const errorJson = JSON.parse(responseText);
          if (errorJson.detail) {
            errorMessage = typeof errorJson.detail === 'string' 
              ? errorJson.detail 
              : JSON.stringify(errorJson.detail);
          } else if (errorJson.error) {
            errorMessage = errorJson.error;
          }
        } catch (e) {
          errorMessage = responseText || errorMessage;
        }
        throw new Error(errorMessage);
      }
      
      try {
        return JSON.parse(responseText);
      } catch (e) {
        console.error("Failed to parse response:", e);
        throw new Error("Invalid response from server");
      }
    } catch (error) {
      console.error("Yield Prediction API Error:", error);
      if (error instanceof Error) {
        throw error;
      }
      throw new Error("An unknown error occurred");
    }
  },

  async diseaseDetection(file: File) {
    try {
      console.log("Sending disease detection request to:", `${API_BASE_URL}/disease-detection`);
      
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch(`${API_BASE_URL}/disease-detection`, {
        method: 'POST',
        body: formData,
      });
      
      console.log("Response status:", response.status);
      const responseText = await response.text();
      console.log("Raw response:", responseText);
      
      if (!response.ok) {
        let errorMessage = 'Failed to detect disease';
        try {
          const errorJson = JSON.parse(responseText);
          if (errorJson.detail) {
            errorMessage = typeof errorJson.detail === 'string' 
              ? errorJson.detail 
              : JSON.stringify(errorJson.detail);
          } else if (errorJson.error) {
            errorMessage = errorJson.error;
          }
        } catch (e) {
          errorMessage = responseText || errorMessage;
        }
        throw new Error(errorMessage);
      }
      
      try {
        return JSON.parse(responseText);
      } catch (e) {
        console.error("Failed to parse response:", e);
        throw new Error("Invalid response from server");
      }
    } catch (error) {
      console.error("Disease Detection API Error:", error);
      if (error instanceof Error) {
        throw error;
      }
      throw new Error("An unknown error occurred");
    }
  }
}; 