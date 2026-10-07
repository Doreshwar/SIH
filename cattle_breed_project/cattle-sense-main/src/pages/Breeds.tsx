import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Search, Filter, Eye } from 'lucide-react';
import { mockBreedData } from '@/lib/mockData';

const Breeds = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedUse, setSelectedUse] = useState<string>('');

  const filteredBreeds = mockBreedData.filter(breed => {
    const matchesSearch = breed.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         breed.characteristics.origin.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesUse = !selectedUse || breed.characteristics.primary_use === selectedUse;
    return matchesSearch && matchesUse;
  });

  const uses = [...new Set(mockBreedData.map(breed => breed.characteristics.primary_use))];

  return (
    <div className="min-h-screen gradient-subtle">
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold text-foreground mb-4">
              Cattle Breed Database
            </h1>
            <p className="text-lg text-muted-foreground">
              Explore our comprehensive catalog of cattle breeds
            </p>
          </div>

          {/* Search and Filter */}
          <Card className="mb-8 shadow-card">
            <CardContent className="p-6">
              <div className="flex flex-col md:flex-row gap-4">
                <div className="flex-1 relative">
                  <Search className="absolute left-3 top-3 w-4 h-4 text-muted-foreground" />
                  <Input
                    placeholder="Search breeds, origins, or characteristics..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10"
                  />
                </div>
                <div className="flex gap-2">
                  <Button
                    variant={selectedUse === '' ? 'farm' : 'outline'}
                    onClick={() => setSelectedUse('')}
                    size="sm"
                  >
                    All Uses
                  </Button>
                  {uses.map(use => (
                    <Button
                      key={use}
                      variant={selectedUse === use ? 'farm' : 'outline'}
                      onClick={() => setSelectedUse(use)}
                      size="sm"
                    >
                      {use}
                    </Button>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Results Count */}
          <div className="mb-6">
            <p className="text-muted-foreground">
              Showing {filteredBreeds.length} of {mockBreedData.length} breeds
            </p>
          </div>

          {/* Breed Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredBreeds.map((breed) => (
              <Card key={breed.id} className="shadow-card hover:shadow-elegant transition-smooth overflow-hidden">
                <div className="aspect-video bg-muted">
                  <img
                    src={breed.image}
                    alt={breed.name}
                    className="w-full h-full object-cover"
                  />
                </div>
                <CardHeader className="pb-3">
                  <CardTitle className="text-xl">{breed.name}</CardTitle>
                  <div className="flex gap-2">
                    <Badge variant="secondary">{breed.characteristics.primary_use}</Badge>
                    <Badge variant="outline">{breed.characteristics.origin}</Badge>
                  </div>
                </CardHeader>
                <CardContent className="pt-0">
                  <div className="space-y-3">
                    <div>
                      <p className="text-sm font-medium text-foreground">Average Weight</p>
                      <p className="text-sm text-muted-foreground">{breed.characteristics.average_weight}</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium text-foreground">Distinctive Features</p>
                      <p className="text-sm text-muted-foreground">{breed.characteristics.distinctive_features}</p>
                    </div>
                    <div className="pt-2">
                      <Button variant="outline" size="sm" className="w-full">
                        <Eye className="w-4 h-4" />
                        View Details
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* No Results */}
          {filteredBreeds.length === 0 && (
            <div className="text-center py-12">
              <p className="text-lg text-muted-foreground">
                No breeds found matching your criteria.
              </p>
              <Button
                variant="outline"
                onClick={() => {
                  setSearchTerm('');
                  setSelectedUse('');
                }}
                className="mt-4"
              >
                Clear Filters
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Breeds;