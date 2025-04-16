import React, { useState } from 'react';
import { api } from '../lib/api';

export const FertilizerClassification: React.FC = () => {
  const [choice, setChoice] = useState<'1' | '2' | null>(null);
  const [formData, setFormData] = useState({
    Soil_color: '',
    Nitrogen: '',
    Phosphorus: '',
    Potassium: '',
    pH: '',
    Temperature: '',
    Crop: '',
    Moisture: '',
    Rainfall: '',
    Carbon: '',
    Soil: ''
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
      const response = await api.fertilizerClassification({
        ...formData,
        choice
      });
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
      {!choice ? (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold">Choose Fertilizer Classification Method</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <button
              onClick={() => setChoice('1')}
              className="p-4 border rounded-lg hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <h4 className="font-medium">Based on Soil Color</h4>
              <p className="text-sm text-gray-600">Uses soil color and nutrient levels</p>
            </button>
            <button
              onClick={() => setChoice('2')}
              className="p-4 border rounded-lg hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <h4 className="font-medium">Based on Soil Type</h4>
              <p className="text-sm text-gray-600">Uses soil type and environmental factors</p>
            </button>
          </div>
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="space-y-4">
          {choice === '1' ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">Soil Color</label>
                <select
                  name="Soil_color"
                  value={formData.Soil_color}
                  onChange={handleChange}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  required
                >
                  <option value="">Select Color</option>
                  <option value="Red">Red</option>
                  <option value="Black">Black</option>
                  <option value="Brown">Brown</option>
                  <option value="Yellow">Yellow</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">Nitrogen (ppm)</label>
                <input
                  type="number"
                  name="Nitrogen"
                  value={formData.Nitrogen}
                  onChange={handleChange}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">Phosphorus (ppm)</label>
                <input
                  type="number"
                  name="Phosphorus"
                  value={formData.Phosphorus}
                  onChange={handleChange}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">Potassium (ppm)</label>
                <input
                  type="number"
                  name="Potassium"
                  value={formData.Potassium}
                  onChange={handleChange}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">pH</label>
                <input
                  type="number"
                  name="pH"
                  value={formData.pH}
                  onChange={handleChange}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">Temperature (°C)</label>
                <input
                  type="number"
                  name="Temperature"
                  value={formData.Temperature}
                  onChange={handleChange}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  required
                />
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
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">Temperature (°C)</label>
                <input
                  type="number"
                  name="Temperature"
                  value={formData.Temperature}
                  onChange={handleChange}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">Moisture (%)</label>
                <input
                  type="number"
                  name="Moisture"
                  value={formData.Moisture}
                  onChange={handleChange}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">Rainfall (mm)</label>
                <input
                  type="number"
                  name="Rainfall"
                  value={formData.Rainfall}
                  onChange={handleChange}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">pH</label>
                <input
                  type="number"
                  name="pH"
                  value={formData.pH}
                  onChange={handleChange}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">Nitrogen (ppm)</label>
                <input
                  type="number"
                  name="Nitrogen"
                  value={formData.Nitrogen}
                  onChange={handleChange}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">Phosphorus (ppm)</label>
                <input
                  type="number"
                  name="Phosphorus"
                  value={formData.Phosphorus}
                  onChange={handleChange}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">Potassium (ppm)</label>
                <input
                  type="number"
                  name="Potassium"
                  value={formData.Potassium}
                  onChange={handleChange}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">Carbon (%)</label>
                <input
                  type="number"
                  name="Carbon"
                  value={formData.Carbon}
                  onChange={handleChange}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">Soil Type</label>
                <select
                  name="Soil"
                  value={formData.Soil}
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
            </div>
          )}
          <div className="flex space-x-2">
            <button
              type="button"
              onClick={() => setChoice(null)}
              className="px-4 py-2 border rounded-lg hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              Back
            </button>
            <button
              type="submit"
              disabled={isLoading}
              className="flex-1 bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
            >
              {isLoading ? 'Processing...' : 'Get Recommendation'}
            </button>
          </div>
        </form>
      )}

      {result && (
        <div className="mt-4 p-4 bg-gray-50 rounded-lg">
          <h3 className="text-lg font-semibold mb-2">Recommendation</h3>
          <p className="text-gray-700">{result.message}</p>
        </div>
      )}
    </div>
  );
}; 