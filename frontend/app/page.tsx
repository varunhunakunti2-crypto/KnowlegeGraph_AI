export default function Dashboard() {
  return (
    <div className="p-8">
      <h1 className="text-2xl font-semibold text-gray-900 mb-6">Dashboard</h1>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h3 className="text-sm font-medium text-gray-500">Documents</h3>
          <p className="text-2xl font-semibold text-gray-900 mt-2">-</p>
        </div>
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h3 className="text-sm font-medium text-gray-500">Indexed</h3>
          <p className="text-2xl font-semibold text-gray-900 mt-2">-</p>
        </div>
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h3 className="text-sm font-medium text-gray-500">Status</h3>
          <p className="text-sm text-gray-600 mt-2">Loading...</p>
        </div>
      </div>
    </div>
  );
}
