export interface BreedData {
  id: string;
  name: string;
  image: string;
  characteristics: {
    origin: string;
    primary_use: string;
    average_weight: string;
    distinctive_features: string;
  };
}

// Updated breed data to match the trained ML model
// These are Indian indigenous cattle and buffalo breeds
export const mockBreedData: BreedData[] = [
  {
    id: 'gir',
    name: 'Gir',
    image: '/placeholder-cattle.jpg',
    characteristics: {
      origin: 'Gujarat, India',
      primary_use: 'Milk production',
      average_weight: '350-450 kg',
      distinctive_features: 'Red/white color, curved horns, good milk yield'
    }
  },
  {
    id: 'ongole',
    name: 'Ongole',
    image: '/placeholder-cattle.jpg',
    characteristics: {
      origin: 'Andhra Pradesh, India',
      primary_use: 'Draft and beef',
      average_weight: '500-700 kg',
      distinctive_features: 'White/grey color, large size, prominent hump'
    }
  },
  {
    id: 'murrah',
    name: 'Murrah',
    image: '/placeholder-cattle.jpg',
    characteristics: {
      origin: 'Haryana, India',
      primary_use: 'Milk production',
      average_weight: '500-650 kg',
      distinctive_features: 'Black color, curved horns, high milk yield'
    }
  },
  {
    id: 'jaffarabadi',
    name: 'Jaffarabadi',
    image: '/placeholder-cattle.jpg',
    characteristics: {
      origin: 'Gujarat, India',
      primary_use: 'Milk and draft',
      average_weight: '500-700 kg',
      distinctive_features: 'Large size, curved horns, black color'
    }
  },
  {
    id: 'niliravi',
    name: 'Nili Ravi',
    image: '/placeholder-cattle.jpg',
    characteristics: {
      origin: 'Punjab, Pakistan',
      primary_use: 'Milk production',
      average_weight: '450-550 kg',
      distinctive_features: 'Black color, white markings, curved horns'
    }
  },
  {
    id: 'kankrej',
    name: 'Kankrej',
    image: '/placeholder-cattle.jpg',
    characteristics: {
      origin: 'Gujarat, India',
      primary_use: 'Draft and milk',
      average_weight: '450-550 kg',
      distinctive_features: 'Silver grey color, lyra-shaped horns'
    }
  },
  {
    id: 'vechur',
    name: 'Vechur',
    image: '/placeholder-cattle.jpg',
    characteristics: {
      origin: 'Kerala, India',
      primary_use: 'Milk production',
      average_weight: '150-200 kg',
      distinctive_features: 'Very small size, brown/red color, rare breed'
    }
  },
  {
    id: 'hallikar',
    name: 'Hallikar',
    image: '/placeholder-cattle.jpg',
    characteristics: {
      origin: 'Karnataka, India',
      primary_use: 'Draft work',
      average_weight: '350-450 kg',
      distinctive_features: 'Grey color, medium size, good draught power'
    }
  },
  {
    id: 'amritmahal',
    name: 'Amritmahal',
    image: '/placeholder-cattle.jpg',
    characteristics: {
      origin: 'Karnataka, India',
      primary_use: 'Draft work',
      average_weight: '400-500 kg',
      distinctive_features: 'Grey color, compact body, excellent draught'
    }
  },
  {
    id: 'banni',
    name: 'Banni',
    image: '/placeholder-cattle.jpg',
    characteristics: {
      origin: 'Gujarat, India',
      primary_use: 'Milk production',
      average_weight: '400-500 kg',
      distinctive_features: 'Grey to black color, compact body'
    }
  }
];

// API configuration
export const API_BASE_URL = 'http://localhost:5000';

// Helper function to get breed data by ID
export const getBreedById = (id: string): BreedData | undefined => {
  return mockBreedData.find(breed => breed.id === id);
};

// All breeds available in the model
export const ALL_BREEDS = [
  'toda', 'banni', 'jaffarabadi', 'mehsana', 'murrah', 'nagpuri',
  'niliravi', 'bargur', 'bhadawari', 'ongole', 'gir', 'kankrej',
  'amritmahal', 'hallikar', 'krishnavalley', 'vechur', 'nimari'
];