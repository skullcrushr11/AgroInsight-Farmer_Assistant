import React, { useState } from 'react';
import { api } from '../lib/api';

export const YieldPrediction: React.FC = () => {
  const [formData, setFormData] = useState({
    Soil_Type: '',
    Crop: '',
    Rainfall_mm: '',
    Temperature_Celsius: '',
    Fertilizer_Used: '',
    Irrigation_Used: '',
    Weather_Condition: '',
    Days_to_Harvest: ''
  });
  const [result, setResult] = useState<{ prediction: string; message: string } | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      const response = await api.yieldPrediction(formData);
      setResult(response);
    } catch (error) {
      console.error('Error:', error);
      setResult({
        prediction: 'Error',
        message: 'Sorry, there was an error processing your request.'
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700">Soil Type</label>
            <select
              name="Soil_Type"
              value={formData.Soil_Type}
              onChange={handleChange}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              required
            >
              <option value="">Select Type</option>
              <option value="Clay">Clay</option>
              <option value="Sandy">Sandy</option>
              <option value="Loamy">Loamy</option>
              <option value="Black">Black</option>
              <option value="Red">Red</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Crop</label>
            <input
              type="text"
              name="Crop"
              value={formData.Crop}
              onChange={handleChange}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Rainfall (mm)</label>
            <input
              type="number"
              name="Rainfall_mm"
              value={formData.Rainfall_mm}
              onChange={handleChange}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Temperature (°C)</label>
            <input
              type="number"
              name="Temperature_Celsius"
              value={formData.Temperature_Celsius}
              onChange={handleChange}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Fertilizer Used</label>
            <select
              name="Fertilizer_Used"
              value={formData.Fertilizer_Used}
              onChange={handleChange}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              required
            >
              <option value="">Select Option</option>
              <option value="1">Yes</option>
              <option value="0">No</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Irrigation Used</label>
            <select
              name="Irrigation_Used"
              value={formData.Irrigation_Used}
              onChange={handleChange}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              required
            >
              <option value="">Select Option</option>
              <option value="1">Yes</option>
              <option value="0">No</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Weather Condition</label>
            <select
              name="Weather_Condition"
              value={formData.Weather_Condition}
              onChange={handleChange}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              required
            >
              <option value="">Select Condition</option>
              <option value="Sunny">Sunny</option>
              <option value="Cloudy">Cloudy</option>
              <option value="Rainy">Rainy</option>
              <option value="Windy">Windy</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Days to Harvest</label>
            <input
              type="number"
              name="Days_to_Harvest"
              value={formData.Days_to_Harvest}
              onChange={handleChange}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              required
            />
          </div>
        </div>
        <button
          type="submit"
          disabled={isLoading}
          className="w-full bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
        >
          {isLoading ? 'Processing...' : 'Predict Yield'}
        </button>
      </form>

      {result && (
        <div className="mt-4 p-4 bg-gray-50 rounded-lg">
          <h3 className="text-lg font-semibold mb-2">Prediction Result</h3>
          <p className="text-gray-700">{result.message}</p>
        </div>
      )}
    </div>
  );
}; 